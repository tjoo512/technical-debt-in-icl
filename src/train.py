import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import numpy as np
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler  
from samplers import get_data_sampler, sample_scale
from curriculum import Curriculum
from schema import schema
from models import build_model
from eval import eval_model, load_into_model_from_run
from transformers import get_scheduler
import wandb
from training_utils import compute_and_log_model_norm

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, batch_idx, max_train_steps, 
            k_steps_for_loss="all", num_accum_steps=1, lr_scheduler=None, task_name='none'):
    output = model(xs, ys)

    if k_steps_for_loss == "all":
        loss = loss_func(output, ys)
    else:
        loss = loss_func(
            output[:, -int(k_steps_for_loss) :], ys[:, -int(k_steps_for_loss) :]
        )

    # normalize loss to account for batch accumulation
    loss = loss / num_accum_steps
    loss.backward()

    if ((batch_idx + 1) % num_accum_steps == 0) or (batch_idx + 1 == max_train_steps):
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def wandb_log_task(task, metrics_task, baseline_loss, point_wise_tags, step, suffix="", loss_scaling_factor=1.0):
    wandb.log(
        {
            f"{task}_eval{suffix}/overall_loss": np.mean(metrics_task["mean"]) * loss_scaling_factor,
            f"{task}_eval{suffix}/excess_loss": np.mean(metrics_task["mean"]) * loss_scaling_factor / baseline_loss,
            f"{task}_eval{suffix}/pointwise/loss": dict(
                zip(point_wise_tags, np.array(metrics_task["mean"]) * loss_scaling_factor)
            ),
        },
        step=step,
    )

def get_n_points_eval(task, n_dims, task_kwargs, curriculum):
    return curriculum.n_points_schedule.end


def get_training_optimizer(model, args):
    optimizer = None
    lr_scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    if args.training.schedule is not None:
        assert args.training.schedule == "triangle", "Only triangle learning rate schedule is implemented."
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=args.training.warmup_steps,
            num_training_steps=args.training.train_steps,
        )
    return optimizer, lr_scheduler


def train(model, args):
    optimizer, lr_scheduler = get_training_optimizer(model, args)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"] + 1
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    if args.training.data_transformation_args is not None:
        scale = sample_scale(
            method=args.training.data_transformation_args.get("method", None),
            n_dims=n_dims,
            normalize=args.training.data_transformation_args.get("normalize", False),
            seed=args.training.data_transformation_args.get("seed", None),
        )
    else:
        scale = None

    if args.training.granular_ckpt_till is not None:
        assert args.training.granular_ckpt_every is not None, "granular_ckpt_every cannot be None when granular_ckpt_till is not None."

    if args.training.schedule is not None:
        assert args.training.warmup_steps is not None, "warmup_steps cannot be None when learning rate schedule is not None."

    data_kwargs = args.training.data_kwargs
    if args.training.data == "gaussian":
        if data_kwargs is None:
            data_kwargs = {}
        data_kwargs.update({"scale": scale})

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **data_kwargs)

    excess_tensors = {}
    excess_tensors_eval = {}
    loss_scaling_factor = 1.0

    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        out_dir=args.out_dir,
        is_save_task_pool=args.is_save_task_pool,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))
    # log also when i+1 == args.training.train_steps

    num_training_examples = args.training.num_training_examples
    num_accum_steps = args.training.num_accum_steps
    optimizer.zero_grad()
    log_loss = 0.
    log_point_wise_loss = 0.

    # outputs_list = []
    for i in pbar:
        if (i % num_accum_steps == 0):
            log_loss = 0.
            log_point_wise_loss = torch.zeros(size=(curriculum.n_points,), dtype=torch.float32).cuda()

        data_sampler_args = {}
        task_sampler_args = {}

        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        if "fourier_series" in args.training.task:
            args.training.task_kwargs["max_frequency"] = curriculum.max_freq
            task_sampler = get_task_sampler(
                args.training.task,
                n_dims,
                bsize,
                num_tasks=args.training.num_tasks,
                **args.training.task_kwargs,
            )

        # pdb.set_trace()
        task = task_sampler(**task_sampler_args)

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()
        loss, output = train_step(
            model,
            xs.cuda(),
            ys.cuda(),
            optimizer,
            loss_func,
            batch_idx=i,
            max_train_steps=args.training.train_steps,
            k_steps_for_loss=args.training.k_steps_for_loss,
            num_accum_steps=num_accum_steps,
            lr_scheduler=lr_scheduler,
            task_name=args.training.task,
        )

        log_loss += loss
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)
        point_wise_loss = point_wise_loss/num_accum_steps
        log_point_wise_loss += point_wise_loss

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if (
            (i+1 == num_accum_steps or # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
            (i > 0 and (i+1) % args.wandb.log_every_steps == 0) or # log during training whenever we pass the logging interval
            i+1 == args.training.train_steps) # log at the last train step
            and not args.test_run
        ):
            wandb.log(
                {
                    "overall_loss": log_loss * loss_scaling_factor,
                    "excess_loss": log_loss * loss_scaling_factor / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, log_point_wise_loss.cpu().numpy() * loss_scaling_factor)
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "max_freq": curriculum.max_freq,
                },
                step=(i+1)//num_accum_steps,
            )

        if (
            (i+1 == num_accum_steps or # first log when num_accum_steps are over -- this is equiv. to log at step=0 for non-accumulation training
            (i > 0 and (i+1) % args.training.eval_every_steps == 0) or # log during training whenever we pass the logging interval
            i+1 == args.training.train_steps) # log at the last train step
            and not args.test_run
        ):
            n_dims = args.model.n_dims
            eval_task_sampler_kwargs = args.training.task_kwargs
            eval_data_kwargs = data_kwargs
            metrics = eval_model(
                model,
                task_name=args.training.task,
                data_name=args.training.data,
                n_dims=args.model.n_dims,
                n_points=get_n_points_eval(args.training.task, args.model.n_dims, args.training.task_kwargs, curriculum),
                prompting_strategy="standard",
                batch_size=64,
                data_sampler_kwargs=eval_data_kwargs,
                task_sampler_kwargs=eval_task_sampler_kwargs,
                excess_tensors_eval=excess_tensors_eval
            )

            wandb_log_task(args.training.task, metrics, baseline_loss, point_wise_tags, step=(i+1)//num_accum_steps, loss_scaling_factor=loss_scaling_factor)

            if args.training.log_model_norm:
                compute_and_log_model_norm(model, step=(i+1)//num_accum_steps)

        curriculum.update()

        one_indexed_steps = i + 1
        pbar.set_description(f"loss {loss * loss_scaling_factor}")
        if (one_indexed_steps % args.training.save_every_steps == 0  or one_indexed_steps == args.training.train_steps) and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i, # 0-indexed because this is used while resuming the training
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and ((one_indexed_steps % args.training.keep_every_steps == 0 or one_indexed_steps == args.training.train_steps)
            or (args.training.granular_ckpt_till is not None and one_indexed_steps <= args.training.granular_ckpt_till and one_indexed_steps % args.training.granular_ckpt_every == 0))
            and not args.test_run
            and one_indexed_steps > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{one_indexed_steps}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        f_name = out_dir.split("/")[-1]
        wandb.init(project="tech_debt_icl",
            name=f_name,
            resume=True,
            dir=args.out_dir
        )

    model = build_model(args.model)
    if args.model.load_model_path is not None:
        run_path = os.path.join(args.model.load_model_path)
        load_into_model_from_run(model, run_path)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval

def updateStepsByGradAccumSteps(args):
    num_accum_steps = args.training.num_accum_steps
    args.training.train_steps *= num_accum_steps
    args.training.eval_every_steps *= num_accum_steps
    args.training.save_every_steps *= num_accum_steps
    args.training.keep_every_steps *= num_accum_steps

    args.training.curriculum.dims.interval *= num_accum_steps
    args.training.curriculum.points.interval *= num_accum_steps

    args.wandb.log_every_steps *= num_accum_steps

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")
    updateStepsByGradAccumSteps(args)

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            first_time_train = True
            run_id = str(uuid.uuid4())
        else:
            first_time_train = False
        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            if not first_time_train:
                raise NotImplementedError
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
