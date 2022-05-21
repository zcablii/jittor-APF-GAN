import torch
import numpy as np
import os
import inspect
from pickle import load, dump
import re

def generate_samples(generator, gen_input):
    out = generator.forward(gen_input)
    out = out.cpu().data.numpy()
    return out


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return load(f)


def adjust_dynamic_range(data, range_in, range_out):
    if range_in != range_out:
        (min_in, max_in) = range_in
        (min_out, max_out) = range_out
        scale_factor = (max_out - min_out) / (max_in - min_in)
        data = (data - min_in) * scale_factor + min_out
    return data


def numpy_upsample_nearest(x, n_last_dims, size=None, scale_factor=None):
    try:
        shape = x.shape[-n_last_dims:]
        if size is not None:
            if type(size) is int:
                size = (size,) * n_last_dims
            for i in range(n_last_dims):
                if size[i] % shape[i] != 0:
                    raise Exception('Incompatible sizes: {} and {}.'.format(x.shape, size))
            scale_factor = tuple((target_s // source_s for source_s, target_s in zip(shape, size)))
        if scale_factor is None:
            raise Exception('Either size or scale_factor must be specified.')
        if type(scale_factor) is int:
            scale_factor = (scale_factor,) * n_last_dims
        for i in range(n_last_dims):
            if scale_factor[i] > 1:
                x = x.repeat(scale_factor[i], axis=-n_last_dims + i)
        return x
    except Exception as e:
        print('Args or shapes in numpy_upsample: x {} size {} scale_factor {}'.format(x.shape, size, scale_factor))
        raise e


def random_latents(num_latents, latent_size):
    return torch.from_numpy(np.random.randn(num_latents, latent_size).astype(np.float32))


def create_result_subdir(results_dir, experiment_name):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    path = os.path.join(results_dir, experiment_name)
    os.makedirs(path,exist_ok=True)
    return path


def get_all_classes(module):
    return [getattr(module, name) for name in dir(module)
            if inspect.isclass(getattr(module, name, 0))]


def generic_arg_parse(x, hinttype=None):
    if hinttype is int or hinttype is float or hinttype is str:
        return hinttype(x)
    try:
        for _ in range(2):
            x = x.strip('\'').strip("\"")
        __special_tmp = eval(x, {}, {})
    except:  # the string contained some name - probably path, treat as string
        __special_tmp = x  # treat as string
        print('Treating value: {} as str.'.format(x))
    return __special_tmp


def create_params(classes, excludes=None, overrides=None):
    params = {}
    if not excludes:
        excludes = {}
    if not overrides:
        overrides = {}
    for cls in classes:
        nm = cls.__name__
        params[nm] = {
            k: (v.default if nm not in overrides or k not in overrides[nm] else overrides[nm][k])
            for k, v in dict(inspect.signature(cls.__init__).parameters).items()
            if v.default != inspect._empty and
            (nm not in excludes or k not in excludes[nm])
        }
    return params


def get_structured_params(params):
    new_params = {}
    for p in params:
        if '.' in p:
            [cls, attr] = p.split('.', 1)
            if cls not in new_params:
                new_params[cls] = {}
            new_params[cls][attr] = params[p]
        else:
            new_params[p] = params[p]
    return new_params


def params_to_str(params):
    s = '{\n'
    for k, v in params.items():
        s += '\t\'{}\': {},\n'.format(k, repr(v))
    s += '}'
    return s


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    items.sort(key=natural_keys)



def save_network(net, label, epoch, params):
    result_dir = params['result_dir']

    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(result_dir, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    net.cuda()


def load_network(net, label, epoch, params):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = params['result_dir']
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    print('loaded checkpoint ',save_path )
    return net

# save_network(self.netG, 'G', epoch, self.opt)
# netG = load_network(netG, 'G', opt.which_epoch, opt)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled

