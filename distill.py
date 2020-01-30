import numpy as np
import chainer
import chainer.functions as F
import logging
import copy
#from tqdm import *

def reset_model_params(model):
    for p in model.params():
        p.initializer(p.array)

def copy_model_params(src, dst):
    dst.copyparams(src)
    for src_p, dst_p in zip(src.params(), dst.params()):
        dst_p.update_rule = copy.deepcopy(src_p.update_rule)

def train_bc_batch(model, target_model, loss_fn, train_inputs, batch_size=256,
        n_epochs=1, lr=1e-3, predict_fn=lambda m, x: m(x), with_inputs=False, fix_batch_num=False, num_batch=20):
    device = model.device
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    epoch_losses = []

    for epoch in range(n_epochs):
        n_samples = len(train_inputs)
        if fix_batch_num:
            indices = np.random.randint(n_samples - batch_size, size=num_batch)
        else:
            indices = np.asarray(range(n_samples - batch_size))[::batch_size]
            np.random.shuffle(indices)

        losses = []
        logging.info('Num batches: {}'.format(len(indices)))
        for start_idx in indices:
            #batch_idx = indices[start_idx:start_idx + batch_size].astype(np.int32)
            batch_idx = list(range(start_idx, start_idx + batch_size))
            inputs_batch = train_inputs[batch_idx]
            inputs_batch = chainer.Variable(inputs_batch)
            inputs_batch.to_device(device)
            targets_batch = predict_fn(target_model, inputs_batch)
            preds_batch = predict_fn(model, inputs_batch)

            model.cleargrads()
            if with_inputs:
                loss = loss_fn(inputs_batch, preds_batch, targets_batch)
            else:
                loss = loss_fn(preds_batch, targets_batch)
            loss.backward()
            optimizer.update()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        epoch_losses.append(mean_loss)
    return epoch_losses

def train_bc(model, loss_fn, train_inputs, train_targets, batch_size,
        n_epochs=1, lr=1e-3, predict_fn=lambda m, x: m(x), loss_fn_with_inputs=False):

    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)
    epoch_losses = []

    for epoch in range(n_epochs):
        n_samples = len(train_inputs)
        indices = np.array(range(n_samples))
        np.random.shuffle(indices)
        losses = []
        for start_idx in (range(0, n_samples, batch_size)):
            batch_idx = indices[start_idx:start_idx + batch_size].astype(np.int32)
            inputs_batch = train_inputs[batch_idx]
            targets_batch = train_targets[batch_idx]
            preds_batch = predict_fn(model, inputs_batch)

            model.cleargrads()
            if loss_fn_with_inputs:
                loss = loss_fn(inputs_batch, preds_batch, targets_batch)
            else:
                loss = loss_fn(preds_batch, targets_batch)
            loss.backward()
            optimizer.update()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        epoch_losses.append(mean_loss)
    return epoch_losses

def distill(teacher, student, loss_fn, train_inputs, batch_size, n_epochs=1, lr=1e-3, predict_fn=lambda m, x: m(x)):
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(student)
    epoch_losses = []

    train_targets = predict_fn(teacher, train_inputs)

    for epoch in range(n_epochs):
        n_samples = len(train_inputs)
        indices = np.array(range(n_samples))
        np.random.shuffle(indices)
        losses = []
        for start_idx in tqdm(range(0, n_samples, batch_size)):
            batch_idx = indices[start_idx:start_idx + batch_size].astype(np.int32)
            inputs_batch = train_inputs[batch_idx]
            targets_batch = train_targets[batch_idx]
            preds_batch = predict_fn(student, inputs_batch)

            student.cleargrads()
            loss = loss_fn(preds_batch, targets_batch)
            loss.backward()
            optimizer.update()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        epoch_losses.append(mean_loss)
    return epoch_losses

if __name__ == '__main__':
    from chainer import Function, Variable, optimizers, serializers, utils, initializers
    from chainer import Link, Chain, ChainList
    import chainer.links as L
    import time

    class teacherNet(chainer.Chain):
        def __init__(self):
            super(teacherNet, self).__init__(
                fc1 = L.Linear(28 * 28, 800),
                fc2 = L.Linear(800, 800),
                fc3 = L.Linear(800, 10),
            )

        def __call__(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class studentNet(chainer.Chain):
        def __init__(self):
            super(studentNet, self).__init__(
                fc1 = L.Linear(28 * 28, 8),
                fc2 = L.Linear(8, 8),
                fc3 = L.Linear(8, 10),
            )
        def __call__(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    train, test = chainer.datasets.get_mnist()
    n_cls = 10
    train_num = len(train)
    test_num = len(test)
    print('train num:%d, test num:%d' % (train_num, test_num))

    train_iter = chainer.iterators.SerialIterator(train, 256)
    test_iter = chainer.iterators.SerialIterator(test, 1000, repeat=False, shuffle=False)

    def train_model(model, optimizer, train_iter, test_iter, max_iters, out_dir, out_name):
        ts = time.strftime("%Y%m%d%H%M",time.localtime())
        model.predictor.train = True

        # begin training
        for iters in range(max_iters):
            model.predictor.train = True
            batch = train_iter.next()
            x_data = np.array([x[0] for x in batch])
            t = np.array([x[1] for x in batch])
            x_data = Variable(x_data)
            t = Variable(t)
            loss = model(x_data,t)
            accuracy = F.accuracy(model.predictor(x_data), t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            if (iters + 1) % (max_iters // 2) ==0:
                optimizer.lr *= 0.5
            test_accuracy = cal_accuracy(model, test_iter)
            txt_log = '(%s) Iteration:%d   Loss:%f   Training Accuracy:%f   Test Accuracy:%f' % (out_name, iters + 1, loss.data, accuracy.data, test_accuracy)
            print(txt_log)

    def cal_accuracy(model,test_iter):
        model.predictor.train = False
        acc = []
        test_iter.reset()
        for i,batch in enumerate(test_iter):
            x_data = np.array([x[0] for x in batch])
            t = np.array([x[1] for x in batch])
            x_data = Variable(x_data)
            t = Variable(t)
            acc.append(F.accuracy(model.predictor(x_data),t).data)
        return sum(acc)/len(acc)

    max_iters = 2
    learning_rate = 0.01

    teacher_model = L.Classifier(teacherNet())
    optimizer = optimizers.MomentumSGD(lr = learning_rate)
    optimizer.use_cleargrads()
    optimizer.setup(teacher_model)

    train_model(teacher_model, optimizer, train_iter, test_iter, max_iters, None, 'teacher')

    student_model = studentNet()
    teacher_model.predictor.train = False
    train_data = np.array([d[0] for d in train])
    train_data = Variable(train_data)
    distill(teacher_model.predictor, student_model, loss_fn=F.softmax_cross_entropy, train_inputs=train_data, batch_size=1024)
