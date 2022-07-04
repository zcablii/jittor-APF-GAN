import torch
import misc
import torch.utils.data


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode == "nori":
        return "NoriDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    if opt.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train,
                                                            num_replicas=num_tasks,
                                                            rank=global_rank,
                                                            shuffle=True)
        if opt.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val,
                                                              num_replicas=num_tasks,
                                                              rank=global_rank,
                                                              shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=opt.batch_size//num_tasks, drop_last=True, num_workers=opt.num_workers, pin_memory=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=opt.batch_size//num_tasks, drop_last=False, num_workers=opt.num_workers, pin_memory=True)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True, num_workers = opt.num_workers)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers = opt.num_workers)

    return dataloader_train, dataloader_val