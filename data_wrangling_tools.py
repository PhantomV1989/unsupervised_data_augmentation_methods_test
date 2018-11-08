import os
import numpy as np
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from scipy.stats import chi

import matplotlib.pyplot as plt
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_sliding_window(data, window_size, interval=1, peek_ahead=0):
    # if peek>0, returns the window with its peeked data pair
    last = window_size + peek_ahead
    new_data = []
    for i in range(0, len(data) - last + 1, interval):
        data_point = data[i:i + window_size] if peek_ahead == 0 else [data[i:i + window_size],
                                                                      data[
                                                                      i + window_size:i + window_size + peek_ahead]]
        new_data.append(data_point)
    return new_data


def maxpool_sampling(data, sampling_interval):
    new_data = []
    for i in range(0, len(data), sampling_interval):
        temp = data[i:i + sampling_interval]
        new_data.append(max(temp))
    return new_data


def split_data(X, Y, split_ratio):  # 0.000026s
    if len(X) != len(Y):
        print('Data set not of same size')
        return
    dLen = len(X)
    aLen = int(dLen * split_ratio)
    output = (X[:aLen], Y[:aLen]), (X[aLen:], Y[aLen:])
    return output


def random_sampling(X, Y, sample_size, replacement=True):
    X_new, Y_new = [], []
    if replacement:
        while len(X_new) < sample_size:
            rng = np.random.randint(low=0, high=len(X))
            X_new.append(X[rng])
            Y_new.append(Y[rng])
    else:
        if sample_size > len(X):
            print('Sample size greater than population size')
            return
        else:
            while len(X_new) < sample_size:
                rng = np.random.randint(low=0, high=len(X))
                X_new.append(X.pop(rng))
                Y_new.append(Y.pop(rng))
    return X_new, Y_new


def _get_bin_pos(v, bin_rng, return_bin_range=False):
    if v < bin_rng[0]:
        return [0, [bin_rng[0], bin_rng[1]]] if return_bin_range else 0
    for i in range(len(bin_rng) - 1):
        if v >= bin_rng[i] and v < bin_rng[i + 1]:
            return [i, [bin_rng[i], bin_rng[i + 1]]] if return_bin_range else i
    return [i, [bin_rng[-2], bin_rng[-1]]] if return_bin_range else i


def transform_data_to_uniform_dist(data, bins=30, retain_size=True):
    count, ranges = np.histogram(data, bins=bins)
    count = [(1 if v == 0 else v) for v in count]
    multiplier = [int(x) for x in np.rint(max(count) / count)]

    new_data = []
    for j in data:
        pos = _get_bin_pos(j, ranges)
        mult = multiplier[pos]
        temp = np.full((mult, 1), j)

        if new_data == []:
            new_data = np.ndarray.copy(temp)
        else:
            new_data = np.concatenate((new_data, temp))

    # count, ranges = np.histogram(new_data, bins=bins)
    # for debugging, make sure count is evenly distributed
    if retain_size:
        np.random.shuffle(new_data)
        return new_data[:len(data)]
    return new_data


def transform_1D_data_to_reverse_dist(data, no_of_new_samples=0, return_same_sized_combined_dist=True, bins=30,
                                      imba_f=1.2,
                                      show_visualization=True):
    # instead of making rare events having the same standing as frequent events, we make rare events even more common than norm
    # imba factor controls the distribution of rare events > normal events

    # if no_of_new_samples is not specified, it attempts to calculate the number by finding the amount of new samples
    # required to fill up the remaining area of the uniform dist (think of it as the unfilled area of a rectangle'

    latent_dim = 1
    feature_count = len(data[0])
    enc_input = Input(shape=(feature_count,))

    encoder = Sequential()
    encoder.add(Dense(100, input_shape=(feature_count,)))
    encoder.add(Dense(latent_dim))

    decoder = Sequential()
    decoder.add(Dense(100, input_shape=(latent_dim,)))
    decoder.add(Dense(feature_count))

    final = Model(enc_input, decoder(encoder(enc_input)))
    final.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")

    np.random.shuffle(data)

    final.fit(x=np.asarray(data), y=np.asarray(data), batch_size=int(len(data) / 10),
              callbacks=[EarlyStopping(monitor='loss', min_delta=0.00001)],
              epochs=500)

    latent_values = encoder.predict(data)

    if show_visualization:
        plt.figure('Original latent values histogram')
        plt.hist(latent_values, bins=bins)

    if bins > len(latent_values):
        bins = int(len(latent_values) / 2)
    count, ranges = np.histogram(latent_values, bins=bins)

    if no_of_new_samples == 0:
        no_of_new_samples = np.sum(np.max(count) - count)

    bins_probability_table = [np.power(x, imba_f) for x in np.rint(max(count) - count) / max(count)]
    bins_probability_table /= np.max(bins_probability_table)

    new_latent_values = []

    while (True):
        for i in range(len(bins_probability_table)):
            bin_rng = [ranges[i], ranges[i + 1]]
            bins_prob = bins_probability_table[i]
            if np.random.rand() < bins_prob:
                new_synth_latent = np.random.rand() * (bin_rng[1] - bin_rng[0]) + bin_rng[0]
                new_latent_values.append([new_synth_latent])
            if len(new_latent_values) >= no_of_new_samples:
                break
        if len(new_latent_values) >= no_of_new_samples:
            break

    if show_visualization:
        plt.figure('New latent values histogram')
        plt.hist(np.asarray(new_latent_values), bins=bins)

    '''
    new_latent_values = []
    for i in range(len(latent_values_bins_pos)):
        bin_pos, bin_rng = latent_values_bins_pos[i]
        mx = multiplier[bin_pos]
        for j in range(mx):

            new_synth_latent = np.random.rand() * (bin_rng[1] - bin_rng[0]) + bin_rng[0]
            new_latent_values.append([new_synth_latent])
    '''
    # for debugging
    if len(new_latent_values) == 0:
        return data
    new_synth_data = decoder.predict(np.asarray(new_latent_values))

    if show_visualization:
        plt.figure('New latent values histogram')
        plt.hist(np.asarray(new_latent_values), bins=bins)

    if show_visualization:
        plt.figure('Combined latent values histogram')
        combined_latent_values = np.concatenate((np.asarray(new_latent_values), latent_values))
        np.random.shuffle(combined_latent_values)

        plt.hist(combined_latent_values, bins=bins)
        plt.show()

    # count_, ranges_ = np.histogram(new_latent_values, bins=bins)

    if return_same_sized_combined_dist == True:
        resampled_data = np.concatenate((data, new_synth_data))
        np.random.shuffle(resampled_data)
        resampled_data = resampled_data[:len(data)]

        # for debugging
        # debugging_latent_v = encoder.predict(resampled_data)
        # plt.hist(debugging_latent_v, bins=bins)
        # plt.show()

        return resampled_data
    return new_latent_values


def transform_1D_samples_using_DOPE(data, return_same_sized_combined_dist=True, new_sample_ratio=0.3, no_of_std=3):
    latent_dim = 1
    no_of_new_samples = int(len(data) * new_sample_ratio)
    feature_count = len(data[0])
    enc_input = Input(shape=(feature_count,))

    encoder = Sequential()
    encoder.add(Dense(100, input_shape=(feature_count,)))
    encoder.add(Dense(latent_dim))

    decoder = Sequential()
    decoder.add(Dense(100, input_shape=(latent_dim,)))
    decoder.add(Dense(feature_count))

    final = Model(enc_input, decoder(encoder(enc_input)))
    final.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")

    np.random.shuffle(data)

    final.fit(x=np.asarray(data), y=np.asarray(data), epochs=50)

    latent_values = encoder.predict(data)

    center = np.mean(latent_values, axis=0)
    std = np.std(latent_values, axis=0)
    chi_std = chi.std(2, 0, np.linalg.norm(std))

    # x-mean
    dist = np.linalg.norm(latent_values - center, axis=1)  # Frobenius norm

    for i, el in enumerate(dist):
        dist[i] = 0. if el > no_of_std * chi_std else dist[i]
    threshold = sorted(dist)[int(len(dist) * 0.9)]
    print('Threshold: ', threshold)

    dist = [0. if x < threshold else x for x in dist]
    dist /= np.sum(dist)

    synth_latent = []
    for i in range(no_of_new_samples):
        # choose an ele from 1st argv, given that 1st argv has prob dist in p
        choice = np.random.choice(np.arange(len(dist)), p=dist)

        a = latent_values[choice]
        latent_copy = np.concatenate((latent_values[:choice], latent_values[choice + 1:]))
        latent_copy -= a
        latent_copy = np.linalg.norm(latent_copy, axis=1)  # Frobenius norm
        b = np.argmin(latent_copy)
        if b >= choice:
            b += 1
        b = latent_values[b]
        scale = np.random.rand()
        c = scale * (a - b) + b
        synth_latent.append(c)

    count, ranges = np.histogram(latent_values, bins=30)
    new_latent_values = np.concatenate((latent_values, np.asarray(synth_latent)))

    count_, ranges_ = np.histogram(new_latent_values, bins=30)

    new_data = decoder.predict(np.asarray(synth_latent))
    if return_same_sized_combined_dist:
        resampled_data = np.concatenate((data, new_data))
        np.random.shuffle(resampled_data)
        return resampled_data[:len(data)]
    return new_data


# tests
def test_x():
    print('Test X')
    arr = []
    s = np.sin

    for i in range(1000):
        v = s(i * 0.2)
        if i % 13 == 0:
            v += np.random.rand()
        if i % 20 == 0:
            v -= np.random.rand()
        if i % np.random.randint(12, 30) == 0:
            v += np.random.rand()
        arr.append(v)
    plt.figure('Sample time series data')
    plt.plot(arr)
    # plt.show()
    np.savetxt('arr.csv', arr, delimiter=',')
    new_arr = create_sliding_window(arr, window_size=10)
    x = np.asarray(new_arr)

    def create_model(x_input):
        model = Sequential()
        model.add(Dense(units=50, input_shape=(x_input.shape[1],)))
        model.add(Dense(units=20))
        model.add(Dense(units=50))
        model.add(Dense(x_input.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        return model

    def get_loss(model, x, y, type='mean'):
        # type = {'mean', 'max'}
        y_ = model.predict(np.asarray(x))
        if type == 'mean':
            return np.mean(np.abs(np.add(y_, -y)), axis=1)
        elif type == 'max':
            return np.asarray([np.max(q) for q in np.abs(np.add(y_, -y))])
        return

    def helper(x):
        model_a = create_model(x)
        np.random.shuffle(x)
        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001)
        history = model_a.fit(x=x, y=x, batch_size=int(len(x) / 10), epochs=500, callbacks=[early_stopping])
        loss = get_loss(model_a, x=x, y=x, type='max')
        return loss

    # original loss
    loss_a = helper(x)

    # loss with optimized 'distribution reversal' sampling
    max_loss_b = []
    best_loss_b = 0

    for i in range(3, 10, 1):
        for j in range(1, 10, 1):
            f = i / 10
            size = int(len(x) * j / 10)
            new_x = transform_1D_data_to_reverse_dist(x, no_of_new_samples=size, return_same_sized_combined_dist=True,
                                                      imba_f=f, show_visualization=False)
            loss_b = helper(np.asarray(new_x))
            max_loss_b.append([f, size, np.max(loss_b)])
            if i == 1:
                best_loss_b = loss_b
            elif np.max(loss_b) <= np.min([x[2] for x in max_loss_b]):
                best_loss_b = loss_b
    loss_b = best_loss_b

    # loss with optimized 'DOPE' sampling
    # may not be optimized efficiently because there is more than just 1 parameter for optimization
    max_loss_c = []
    best_loss_c = 0
    for i in range(1, 10, 1):
        for j in range(1, 4):
            f = i / 10
            std = j
            new_x_doped = transform_1D_samples_using_DOPE(x, return_same_sized_combined_dist=True, new_sample_ratio=f,
                                                          no_of_std=std)
            loss_c = helper(np.asarray(new_x_doped))
            max_loss_c.append([f, std, np.max(loss_c)])
            if i == 1:
                best_loss_c = loss_c
            elif np.max(loss_c) <= np.min([x[2] for x in max_loss_c]):
                best_loss_c = loss_c
    loss_c = best_loss_c

    x = np.arange(len(loss_a))
    plt.figure(2)
    plt.plot(x, loss_a, 'r--', x, loss_b, 'b--', x, loss_c, 'g--')
    plt.show()
    np.savetxt('losses2.csv', np.asarray([loss_a, loss_b, loss_c]), delimiter=',')

    return


if __name__ == '__main__':
    test_x()
