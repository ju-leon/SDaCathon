
import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from meta_learning.get_tasksets import get_tasksets


def accuracy(predictions, targets):
    treshold = torch.tensor([0.5])
    binary_predictions = (predictions.cpu() > treshold)*1
    equal = binary_predictions == targets.cpu()
    sum = equal.type(torch.float).sum()
    n = targets.size(0)
    return sum / n


def meta_train_test_split(data, labels):
    shots = np.min([5, data.size(0) // 2])
    indices = torch.cat([torch.ones(shots, dtype=torch.bool), torch.zeros(
        data.size(0) - shots, dtype=torch.bool)])
    random_permuation = torch.randperm(indices.size(0))
    adaptation_indices = indices[random_permuation]
    evaluation_indices = torch.logical_not(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    return adaptation_data, adaptation_labels, evaluation_data, evaluation_labels


def fast_adapt(adaptation_data, adaptation_labels, evaluation_data, evaluation_labels, learner, loss, adaptation_steps, device):
    adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)
    evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)

    # Adapt the model
    for step in range(adaptation_steps):
        prediction = learner(adaptation_data)
        train_error = loss(prediction.squeeze(), adaptation_labels.squeeze())
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions.squeeze(), evaluation_labels.squeeze())
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    tasksets = get_tasksets()

    # Create model
    # model = l2l.vision.models.OmniglotFC(3 * (28 ** 2), ways)
    model = l2l.vision.models.MiniImagenetCNN(output_size=ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            data, labels = tasksets.train.sample()
            meta_sets = meta_train_test_split(data, labels)
            adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = meta_sets
            evaluation_error, evaluation_accuracy = fast_adapt(
                adaptation_data, adaptation_labels, evaluation_data, evaluation_labels, learner, loss, adaptation_steps, device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            data, labels = tasksets.validation.sample()
            meta_sets = meta_train_test_split(data, labels)
            adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = meta_sets
            evaluation_error, evaluation_accuracy = fast_adapt(
                adaptation_data, adaptation_labels, evaluation_data, evaluation_labels, learner, loss, adaptation_steps, device)

            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error)
        print('Meta Train Accuracy', meta_train_accuracy)
        print('Meta Valid Error', meta_valid_error)
        print('Meta Valid Accuracy', meta_valid_accuracy)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for index in range(len(tasksets.test_adaptation)):
        # Compute meta-testing loss
        learner = maml.clone()
        adaptation_data, adaptation_labels = tasksets.test_adaptation[index]
        evaluation_data, evaluation_labels = tasksets.test_evaluation[index]
        evaluation_error, evaluation_accuracy = fast_adapt(
            adaptation_data, adaptation_labels, evaluation_data, evaluation_labels, learner, loss, adaptation_steps, device)

        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()
