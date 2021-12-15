from my_imports import *
from tools import convert_to_gpu
from generating_data import generate_z, generate_real_data
from defining_models import cal_gradient_penalty, cal_gradient_penalty_generator


def train_discriminator(discriminator, generator, d_optimizer, step, config):
    z = convert_to_gpu(generate_z(config.batch_size, config.z_var, config), config)
    gz = convert_to_gpu(generator(z), config)
    d_fake = discriminator(gz)
    real_data = generate_real_data(config.batch_size, config, config.training_mode)
    if config.disc_type=='simpleReLU' or config.disc_type=='bjorckGroupSort':
        real_data = real_data.view(config.batch_size, -1)
    else:
        real_data = real_data.view(config.batch_size, 1, 28, 28)
    real_data = convert_to_gpu(real_data, config)
    d_real = discriminator(real_data)

    if config.loss_type == 'hinge':
        loss = nn.ReLU()(1.0 + d_fake).mean() + nn.ReLU()(1.0 - d_real).mean()
    elif config.loss_type == 'vanilla':
        ones = convert_to_gpu((torch.FloatTensor([1])).expand_as(d_real), config)
        zeros = convert_to_gpu((torch.FloatTensor([0])).expand_as(d_fake), config)
        loss = nn.BCEWithLogitsLoss()(d_real,ones) + config.BCE(d_fake,zeros)
    elif config.loss_type == 'relativistic':
        ones = convert_to_gpu((torch.FloatTensor([1])).expand_as(d_real), config)
        loss = nn.BCEWithLogitsLoss()(d_real - d_fake,ones)
    elif config.loss_type == 'wgan-gp':
        if config.disc_type == "simpleReLU" or config.disc_type == "mnist":
            gradient_penalty, gradients = cal_gradient_penalty(discriminator, real_data, gz, 0, config = config)
            loss = d_fake.mean() - d_real.mean() + gradient_penalty
        else:
            loss = d_fake.mean() - d_real.mean()
    
    d_optimizer.zero_grad()
    loss.backward()
    d_optimizer.step()


def train_discriminator_with_importance_weights(discriminator, generator, importance_weighter, d_optimizer, step, config):
    z = convert_to_gpu(generate_z(config.batch_size, config.z_var, config), config)
    gz = generator(z)
    wz = importance_weighter(z)
    d_fake = discriminator(gz)
    real_data = generate_real_data(config.batch_size, config, config.training_mode)
    if config.disc_type=='simpleReLU' or config.disc_type=='bjorckGroupSort':
        real_data = real_data.view(config.batch_size, -1)
    else:
        real_data = real_data.view(config.batch_size, 1, 28, 28)
    real_data = convert_to_gpu(real_data, config)
    d_real = discriminator(real_data)

    if config.disc_type == "simpleReLU" or config.disc_type == "mnist":
        gradient_penalty, _ = cal_gradient_penalty(discriminator, real_data, gz, 0, config = config)
        loss = (wz*d_fake).mean() - d_real.mean() + gradient_penalty
    else:
        loss = (wz*d_fake).mean() - d_real.mean()
    
    d_optimizer.zero_grad()
    loss.backward()
    d_optimizer.step()


def train_generator(discriminator, generator, g_optimizer, step, config):
    z = convert_to_gpu(generate_z(config.batch_size, config.z_var, config), config)
    z.requires_grad_(True)
    gz = generator(z)
    d_fake = discriminator(gz)
    
    if config.loss_type == 'hinge':
        loss = - d_fake.mean()
    elif config.loss_type == 'vanilla':
        ones = convert_to_gpu((torch.FloatTensor([1])).expand_as(d_fake), config)
        loss = nn.BCEWithLogitsLoss()(d_fake,ones)
    elif config.loss_type == 'relativistic':
        real_data = convert_to_gpu(generate_real_data(config.batch_size, config, config.training_mode), config)
        d_real = discriminator(real_data)
        ones = convert_to_gpu((torch.FloatTensor([1])).expand_as(d_fake), config)
        loss = nn.BCEWithLogitsLoss()(d_fake - d_real,ones)
    elif config.loss_type == 'wgan-gp':
        loss = - d_fake.mean()
        #gradient_penalty = cal_gradient_penalty_generator(generator, gz, z, constant=2., config=config)
        #loss += gradient_penalty
    
    g_optimizer.zero_grad()
    loss.backward()
    g_optimizer.step()


def train_importance_weighter(discriminator, generator, importance_weighter, iw_optimizer, step, config):
    z = convert_to_gpu(generate_z(config.batch_size, config.z_var, config), config)
    gz = generator(z)
    wz = importance_weighter(z)
    d_fake = discriminator(gz)
    
    if config.loss_type == 'hinge':
        loss = - (wz*d_fake).mean()
    elif config.loss_type == 'vanilla':
        ones = (torch.FloatTensor([1])).expand_as(d_real)
        d_fake = torch.sigmoid(d_fake)
        d_fake_loss = - (wz * torch.log(d_fake)).mean()
        loss = d_fake_loss
    elif config.loss_type == 'relativistic':
        real_data = generate_mixture_gaussian_batch(config.batch_size)
        d_real = discriminator(real_data)
        ones = (torch.FloatTensor([1])).expand_as(d_real)
        loss = nn.BCEWithLogitsLoss()(d_fake - d_real,ones)
    elif config.loss_type == 'wgan-gp':
        loss_emd = (-(wz*(d_fake-min(d_fake))).mean())
    
    loss_reg = config.iw_regul_weight*(torch.pow((wz.mean() - 1), 2) + nn.ReLU()(wz-5).mean())
    loss = loss_emd + loss_reg
    if step%100000==0:
        print("Losses", loss_emd.detach(), loss_reg.detach(), loss.detach())
        print("Weights", wz.mean().detach(), min(wz).detach(), max(wz).detach())
    iw_optimizer.zero_grad()
    loss.backward()
    iw_optimizer.step()


def save_models(net, step, name, config):
    folder = os.path.join(config.name_exp, 'models')
    if not os.path.exists(folder):
        os.makedirs(folder)
    PATH = os.path.join(folder, name+'_'+str(step)+'.pth')
    torch.save(net.state_dict(), PATH)

def load_models(generator, step, config):
    folder = os.path.join(config.name_exp, 'models')
    PATH = os.path.join(folder, 'generator_'+str(step)+'.pth')
    generator.load_state_dict(torch.load(PATH))
