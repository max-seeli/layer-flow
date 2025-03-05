from jax import nn, random, grad, jit, numpy as jnp
import jax
from tqdm import tqdm

jax.config.update("jax_debug_nans", True)

def network_fn(params, x):
    for w, b in params[:-1]:
        x = nn.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    return nn.sigmoid(jnp.dot(x, w) + b)

def representation_at_layer(params, x, layer_idx):
    if layer_idx == 0:
        return x
    
    for i, (w, b) in enumerate(params[:-1]):
        x = nn.tanh(jnp.dot(x, w) + b)
        if i == layer_idx - 1:
            return x
    w, b = params[-1]
    print("Last layer")
    return nn.sigmoid(jnp.dot(x, w) + b)

def representation_from_layer(params, x, layer_idx):
    if layer_idx == len(params):
        return x
    
    for i, (w, b) in enumerate(params[layer_idx:-1]):
        print(i)
        x = nn.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    print("Last layers")
    return nn.sigmoid(jnp.dot(x, w) + b)

def init_network(rng, layer_sizes):
    keys = random.split(rng, len(layer_sizes))
    return [(random.normal(k, (i, o)), random.normal(k, (o,))) for i, o, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]

def loss_fn(params, x, y):
    eps = 1e-7
    y_pred = network_fn(params, x)
    y_pred = jnp.clip(y_pred, eps, 1 - eps) # Clip to avoid log(0)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

def update(params, x, y, gradient_fn, lr=0.01):
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, gradient_fn(params, x, y))]


def accuracy(params, x, y):
    y_pred = network_fn(params, x)
    return jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(y_pred, axis=1))

def train(params, x, y, lr=0.01, num_steps=1000):
    gradient_fn = jit(grad(loss_fn))
    for _ in tqdm(range(num_steps)):
        params = update(params, x, y, gradient_fn, lr)
        print("Loss: ", loss_fn(params, x, y))
        print("Accuracy: ", accuracy(params, x, y))
    return params


if __name__ == '__main__':

    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt
    
    rng = random.PRNGKey(0)
    x, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
    y = nn.one_hot(y, 2)

    params = init_network(rng, [2, 2, 2, 2])
    params = train(params, x, y, lr=1, num_steps=5000)

    # Draw samples and decision boundary
    granularity = 1000
    x1 = jnp.linspace(-1.5, 2.5, granularity)
    x2 = jnp.linspace(-1, 1.5, granularity)


    for representation_layer in range(4):
        X1, X2 = jnp.meshgrid(x1, x2)
        X = jnp.stack([X1.ravel(), X2.ravel()], axis=1)
        Y = representation_from_layer(params, X, representation_layer)
        Y = jnp.argmax(Y, axis=1).reshape(X1.shape)

        x_trans = representation_at_layer(params, x, representation_layer)
        plt.contourf(X1, X2, Y, alpha=0.2)
        plt.scatter(x_trans[:, 0], x_trans[:, 1], c=jnp.argmax(y, axis=1))
        plt.show()
