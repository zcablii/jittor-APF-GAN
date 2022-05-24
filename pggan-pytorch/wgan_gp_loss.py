import torch
from torch.autograd import Variable, grad

mixing_factors = None
grad_outputs = None


def mul_rowwise(a, b):
    s = a.size()
    return (a.view(s[0], -1) * b).view(s)


def calc_gradient_penalty(D, real_data, fake_data, iwass_lambda, iwass_target):
    global mixing_factors, grad_outputs
    if mixing_factors is None or real_data.size(0) != mixing_factors.size(0):
        mixing_factors = torch.cuda.FloatTensor(real_data.size(0), 1)
    mixing_factors.uniform_()

    mixed_data = Variable(mul_rowwise(real_data, 1 - mixing_factors) + mul_rowwise(fake_data, mixing_factors), requires_grad=True)
    mixed_scores = D(mixed_data)
    if grad_outputs is None or mixed_scores.size(0) != grad_outputs.size(0):
        grad_outputs = torch.cuda.FloatTensor(mixed_scores.size())
        grad_outputs.fill_(1.)

    gradients = grad(outputs=mixed_scores, inputs=mixed_data,
                     grad_outputs=grad_outputs,
                     create_graph=True, retain_graph=True,
                     only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - iwass_target) ** 2) * iwass_lambda / (iwass_target ** 2)

    return gradient_penalty


def wgan_gp_D_loss(D, G, real_images, input_semantics,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0,
    return_all      = True):

    D.zero_grad()
    G.zero_grad()

    real_data_v = real_images
    # train with real
    D_real = D(real_data_v)
    D_real_loss = -D_real + D_real ** 2 * iwass_epsilon

    # train with fake
    input_semantics = Variable(input_semantics, volatile=True)  # totally freeze netG
    fake = Variable(G(input_semantics).data)
    inputv = fake
    D_fake = D(inputv)
    D_fake_loss = D_fake

    # train with gradient penalty
    gradient_penalty = calc_gradient_penalty(D, real_data_v.data, fake.data, iwass_lambda, iwass_target)
    gp = gradient_penalty
    # gp.backward()

    D_cost = (D_fake_loss + D_real_loss + gp).mean()
    if return_all:
        return D_cost, D_real_loss, D_fake_loss
    return D_cost


def wgan_gp_G_loss(G, D, input_semantics, need_visual=False):
    G.zero_grad()
    input_semantics = Variable(input_semantics)
    G_new = G(input_semantics)
    D_new = -D(G_new)
    G_cost = D_new.mean()
    generated = None
    if need_visual:
        generated = G_new
    return G_cost, generated