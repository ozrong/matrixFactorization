import mxnet as mx
def get_net(max_user, max_item):
    hidden = 500
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')

    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = 1000)
    user = mx.symbol.Flatten(data = user)
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = 1000)
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    item = mx.symbol.Flatten(data = item)
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred