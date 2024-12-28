import multiprocessing

import torch
from det3d import torchie

from .checkpoint import load_checkpoint


def worker_func(
    model_cls,
    model_kwargs,
    checkpoint,
    dataset,
    data_func,
    gpu_id,
    idx_queue,
    result_queue,
):
    model = model_cls(**model_kwargs)
    load_checkpoint(model, checkpoint, map_location="cpu")
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            result = model(**data_func(data, gpu_id))
            result_queue.put((idx, result))


def parallel_test(
    model_cls, model_kwargs, checkpoint, dataset, data_func, gpus, workers_per_gpu=1
):
    """Parallel testing on multiple GPUs.

    Args:
        model_cls (type): Model class type.
        model_kwargs (dict): Arguments to init the model.
        checkpoint (str): Checkpoint filepath.
        dataset (:obj:`Dataset`): The dataset to be tested.
        data_func (callable): The function that generates model inputs.
        gpus (list[int]): GPU ids to be used.
        workers_per_gpu (int): Number of processes on each GPU. It is possible
            to run multiple workers on each GPU.

    Returns:
        list: Test results.
    """
    ctx = multiprocessing.get_context("spawn")
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * workers_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(
                model_cls,
                model_kwargs,
                checkpoint,
                dataset,
                data_func,
                gpus[i % len(gpus)],
                idx_queue,
                result_queue,
            ),
        )
        for i in range(num_workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
