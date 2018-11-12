import argparse
from copy import copy
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# variable initializer lambda
default_init = lambda shape, loc=0., scale=100.: tf.random_normal(shape=shape, mean=loc, stddev=scale)

# data generator
class DataGenerator:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.yy = tfd.Normal(
            loc=tf.cos(5. * self.x) / (tf.abs(self.x) + 1.),
            scale=0.5 * (tf.sin(self.x) + 1.) + 0.05
        )
    def simulate(self, x=None, nb_observations=None):
        if (x is None) and (nb_observations is not None):
            x = np.random.normal(loc=0., scale=1., size=[nb_observations, 1])
        elif (x is not None):
            pass
        else:
            raise Exception('Must specify either x or nb_observations.')
        with tf.Session() as session:
            y, y_loc, y_scale = session.run([self.yy.sample(), self.yy.loc, self.yy.scale], {self.x: x})
        return x, y, y_loc, y_scale

# dense class
class Dense:
    def __init__(self, nb_in, nb_out, activation=tf.identity, init=default_init):
        self.W = tf.Variable(init([nb_in, nb_out]))
        self.b = tf.Variable(tf.zeros([nb_out]))
        self.activation = activation
    def __call__(self, x):
        return self.activation(tf.matmul(x, self.W) + self.b)
    @property
    def weights(self):
        return [self.W, self.b]
    def freeze(self, session):
        _self = copy(self)
        _self.W = tf.constant(session.run(_self.W), dtype=_self.W.dtype.base_dtype)
        _self.b = tf.constant(session.run(_self.b), dtype=_self.b.dtype.base_dtype)
        return _self

# actnorm class
class ActNorm:
    def __init__(self, nb_in, activation=tf.identity, init=default_init):
        self.v = tf.Variable(tfd.softplus_inverse(tf.ones([nb_in])))
        self.b = tf.Variable(tf.zeros([nb_in]))
        self.activation = activation
    def __call__(self, x):
        return self.activation(tf.nn.softplus(self.v) * x + self.b)
    def initialize(self, session, x):
        mean = tf.reduce_mean(x, axis=0)
        var = tf.reduce_mean(tf.square(x - mean), axis=0)
        inv_stddev = tf.exp(-tf.log(var) / 2.)
        v = tfd.softplus_inverse(inv_stddev)
        b = -mean * inv_stddev
        session.run([self.v.assign(v), self.b.assign(b)])
    @property
    def weights(self):
        return [self.v, self.b]
    def freeze(self, session):
        _self = copy(self)
        _self.v = tf.constant(session.run(_self.v), dtype=_self.v.dtype.base_dtype)
        _self.b = tf.constant(session.run(_self.b), dtype=_self.b.dtype.base_dtype)
        return _self

# model class
class Model:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        _x = x
        for layer in self.layers:
            _x = layer(_x)
        return _x
    def initialize(self, session, x):
        _x = tf.constant(x, dtype=self.weights[0].dtype.base_dtype)
        for layer in self.layers:
            try:
                layer.initialize(session, _x)
            except:
                pass
            _x = layer(_x)
    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
    def freeze(self, session):
        _self = copy(self)
        _self.layers = [layer.freeze(session) for layer in _self.layers]
        return _self

# tensorflow model
class Estimator:
    def __init__(self, nb_members, nb_layers, nb_units_per_layer, nce_scale=1., actnorm=True):
        self.nb_members = nb_members
        self.nb_layers = nb_layers
        self.nb_units_per_layer = nb_units_per_layer
        self.nce_scale = nce_scale
        self.actnorm = actnorm
    def build_tf_graph(self):
        x = tf.placeholder(tf.float32, [None, 1])
        y = tf.placeholder(tf.float32, [None, 1])
        layers_reg_mean, layers_reg_logvar, layers_ood_logit = [], [], []
        if self.actnorm:
            layers_reg_mean.append(ActNorm(1))
            layers_reg_logvar.append(ActNorm(1))
            layers_ood_logit.append(ActNorm(1))
        nb_in = 1
        for l in range(self.nb_layers):
            activation = tf.identity if (l + 1) == self.nb_layers else tf.nn.elu
            nb_out = 1 if (l + 1) == self.nb_layers else self.nb_units_per_layer
            layers_reg_mean.append(Dense(nb_in, nb_out, activation))
            layers_reg_logvar.append(Dense(nb_in, nb_out, activation))
            layers_ood_logit.append(Dense(nb_in, nb_out, activation))
            if self.actnorm:
                layers_reg_mean.append(ActNorm(nb_out))
                layers_reg_logvar.append(ActNorm(nb_out))
                layers_ood_logit.append(ActNorm(nb_out))
            nb_in = nb_out
        reg_mean, reg_logvar = Model(layers_reg_mean), Model(layers_reg_logvar)
        yy_mean, yy_logvar = reg_mean(x), reg_logvar(x)
        yy = tfd.Normal(loc=yy_mean, scale=tf.exp(yy_logvar / 2.))
        reg_loss = -tf.reduce_mean(yy.log_prob(y))
        reg_vars = reg_mean.weights + reg_logvar.weights
        # ood classifier
        ood_logit = Model(layers_ood_logit)
        oo_logit = ood_logit(tf.concat([x, x + tf.random_normal(tf.shape(x), 0., self.nce_scale)], axis=0))
        oo = tfd.Bernoulli(logits=oo_logit)
        ood_loss = -tf.reduce_mean(oo.log_prob(tf.concat([tf.zeros_like(x), tf.ones_like(x)], axis=0)))
        ood_vars = ood_logit.weights
        return x, y, reg_mean, reg_logvar, reg_loss, reg_vars, ood_logit, ood_loss, ood_vars
    def train(self, data_dict, opt, nb_iterations, batch_size, bootstrap=False):
        self.x, self.y, reg_mean, reg_logvar, reg_loss, reg_vars, ood_logit, ood_loss, ood_vars = self.build_tf_graph()
        self.components = [tfd.Normal(
            loc=tf.squeeze(tf.zeros_like(self.x)),
            scale=100. * tf.squeeze(tf.ones_like(self.x))
        )]
        nb_rows = data_dict['x'].shape[0]
        with tf.Session() as session:
            msg = 'Training OOD classifier. Iteration {}/{}. Model loss: {:.4f}.'
            train_op = opt.minimize(loss=ood_loss, var_list=ood_vars)
            session.run(tf.global_variables_initializer())
            ood_logit.initialize(session, data_dict['x'])
            for i in range(nb_iterations):
                ind = np.random.choice(range(nb_rows), batch_size)
                _, _loss = session.run(
                    [train_op, ood_loss],
                    {self.x: data_dict['x'][ind]}
                )
                if (i + 1) % 100 == 0:
                    print(msg.format(i + 1, nb_iterations, _loss))
            ood_prob = tf.nn.sigmoid(ood_logit.freeze(session)(self.x))
            mix_probs = [ood_prob] + self.nb_members * [(1. - ood_prob) * tf.ones_like(ood_prob) / self.nb_members]
            self.cat = tfd.Categorical(probs=tf.concat(mix_probs, axis=1))
            print('--------')
            msg = 'Training regression networks. Ensemble member {}/{}. Iteration {}/{}. Model loss: {:.4f}.'
            train_op = opt.minimize(loss=reg_loss, var_list=reg_vars)
            for m in range(self.nb_members):
                if bootstrap:
                    boot = np.random.choice(range(nb_rows), nb_rows, replace=True).tolist()
                else:
                    boot = range(nb_rows)
                session.run(tf.global_variables_initializer())
                reg_mean.initialize(session, data_dict['x'][boot])
                reg_logvar.initialize(session, data_dict['x'][boot])
                for i in range(nb_iterations):
                    ind = np.random.choice(boot, batch_size)
                    _, _loss = session.run(
                        [train_op, reg_loss],
                        {self.x: data_dict['x'][ind], self.y: data_dict['y'][ind]}
                    )
                    if (i + 1) % 100 == 0:
                        print(msg.format(m + 1, self.nb_members, i + 1, nb_iterations, _loss))
                self.components += [tfd.Normal(
                    loc=tf.squeeze(reg_mean.freeze(session)(self.x)),
                    scale=tf.squeeze(tf.exp(reg_logvar.freeze(session)(self.x) / 2.))
                )]
            self.yy = tfd.Mixture(cat=self.cat, components=self.components)

# main
def main(args):
    # simulate data
    dgp = DataGenerator()
    x_train, y_train, _, _ = dgp.simulate(nb_observations=args.nb_observations)
    # build tensorflow model and train
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    model = Estimator(nb_members=args.nb_members, nb_layers=args.nb_layers, nb_units_per_layer=args.nb_units_per_layer, nce_scale=args.nce_scale, actnorm=args.actnorm)
    model.train(data_dict={'x': x_train, 'y': y_train}, opt=opt, nb_iterations=args.nb_iterations, batch_size=args.batch_size, bootstrap=args.bootstrap)
    # plot results
    with tf.Session() as session:
        x_min, x_max = -7., 7.
        y_min, y_max = -5., 5.
        x_span = np.linspace(x_min, x_max, 100)
        y_span = np.linspace(y_min, y_max, 100)
        x_span, y_span = np.meshgrid(x_span, y_span)
        x_span, y_span = x_span.reshape(-1, 1), y_span.reshape(-1, 1)
        logpdf_true = np.squeeze(session.run(
            dgp.yy.log_prob(dgp.y),
            {dgp.x: x_span, dgp.y: y_span}
        ))
        logpdf_est = session.run(
            model.yy.log_prob(tf.squeeze(model.y)),
            {model.x: x_span, model.y: y_span}
        )
        x_span, y_span = np.squeeze(x_span), np.squeeze(y_span)
        plt.figure(figsize=(28, 10))
        plt.subplot(1, 2, 1)
        plt.title('Actual log density surface')
        plt.tripcolor(x_span, y_span, logpdf_true, shading='gouraud')
        plt.plot(x_train, y_train, 'r.')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.clim(-10, 0)
        plt.subplot(1, 2, 2)
        plt.title('Estimated log density surface')
        plt.tripcolor(x_span, y_span, logpdf_est, shading='gouraud')
        plt.plot(x_train, y_train, 'r.')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.clim(-10, 0)
        plt.tight_layout()
        plt.savefig('log_density.png')
        plt.close()

# set up argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_observations', type=int, default=1000)
    parser.add_argument('--nb_members', type=int, default=5)
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--nb_units_per_layer', type=int, default=100)
    parser.add_argument('--nce_scale', type=float, default=200.)
    parser.add_argument('--actnorm', type=bool, default=True)
    parser.add_argument('--nb_iterations', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--bootstrap', type=bool, default=False)
    args = parser.parse_args()
    main(args)
