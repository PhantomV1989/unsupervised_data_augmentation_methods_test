import os
import numpy as np
from keras.models import Sequential, Model, Input, clone_model
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


def split_data(X, split_ratio):  # 0.000026s
    if len(X) != len(Y):
        print('Data set not of same size')
        return
    dLen = len(X)
    aLen = int(dLen * split_ratio)
    output = (X[:aLen], X[aLen:])
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


def transform_1D_data_to_reverse_dist(data, no_of_new_samples=False, return_same_sized_combined_dist=True, bins=30,
                                      imba_f=1.2,
                                      show_visualization=True):
    # instead of making rare events having the same standing as frequent events, we make rare events even more common than norm
    # imba factor controls the distribution of rare events > normal events

    # if no_of_new_samples is not specified, it attempts to calculate the number by finding the amount of new samples
    # required to fill up the remaining area of the uniform dist (think of it as the unfilled area of a rectangle'
    if not no_of_new_samples and no_of_new_samples != 0:
        no_of_new_samples = np.sum(np.max(count) - count)
    if no_of_new_samples == 0 or imba_f == 0:
        return data

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
    if new_sample_ratio == 0 or no_of_std == 0:
        return data

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


def for_looper(loop_function, loops_start_end_step, tensorify_result=False, _argv=[], _first_loop=True):
    #  loops_start_end_int = [[loop1_start,loop1_end,loop1_int],[loop2_start,loop2_end,loop2_int]]
    #  tensorify result, returns result as a tensor, ONLY WORKS WITH NUMBERS!!
    # DO NOT INITIALIZE argv!!!!, for recursive purposes
    start, end, step = loops_start_end_step[0]
    interval_values = np.arange(start, end, step)

    loop_results = []
    if _argv == []:
        _argv = [0 for x in range(len(loops_start_end_step))]
    for i in interval_values:
        argv_new = np.copy(_argv).astype(float)
        argv_new[len(_argv) - len(loops_start_end_step)] = i
        if len(loops_start_end_step) == 1:
            loop_results.append([argv_new, loop_function(argv_new)])
        else:
            loop_results += for_looper(loop_function=loop_function, loops_start_end_step=loops_start_end_step[1:],
                                       _argv=argv_new, _first_loop=False)
    if _first_loop and tensorify_result:
        tensor_positions_flattened = [x[0] for x in loop_results]
        loop_results_flattened = [x[1] for x in loop_results]

        dim_intervals = [np.arange(x[0], x[1], x[2]) for x in loops_start_end_step]
        steps_count_for_each_dim = [len(x) for x in dim_intervals]
        basic_tensor = np.zeros(steps_count_for_each_dim)
        result_tensor = np.copy(basic_tensor)

        tensor_positions_flattened_mapped = []
        for ele in tensor_positions_flattened:
            tensor_positions_flattened_mapped.append(
                [list(np.where(dim_intervals[i] == e)[0])[0] for i, e in enumerate(ele)])

        for i, mapped_pos in enumerate(tensor_positions_flattened_mapped):
            result_tensor[tuple(mapped_pos)] = loop_results_flattened[i]

        tensor_pos_interval_map = list(
            map(lambda x, y, z: [x, y, z], tensor_positions_flattened_mapped, tensor_positions_flattened,
                loop_results_flattened))
        return result_tensor, tensor_pos_interval_map

    return loop_results


class Hidden:
    @staticmethod
    def get_bin_pos(v, bin_rng, return_bin_range=False):
        if v < bin_rng[0]:
            return [0, [bin_rng[0], bin_rng[1]]] if return_bin_range else 0
        for i in range(len(bin_rng) - 1):
            if bin_rng[i] <= v < bin_rng[i + 1]:
                return [i, [bin_rng[i], bin_rng[i + 1]]] if return_bin_range else i
        return [i, [bin_rng[-2], bin_rng[-1]]] if return_bin_range else i

    @staticmethod
    def parse_numpy_where_results(np_where_results):
        return np.asarray(np_where_results).transpose()[0]


class Optimisers:
    @staticmethod
    def n_ary_search_optimization(score_function, search_space_argvs, search_resolution=3, convergence_limit=0.01,
                                  max_iteration=3,
                                  opti_obj='min', search_beyond_original_space=False):
        # score_function MUST return a dict with key 'score'
        # search space is an array of max,mins to used with score_function, eg [[x1min,x1max],[x2min,x2max]]
        # type is 'AUC' if wants a cluster of best extremas or 'best' for a single best extrema

        if opti_obj == 'min':
            obj_fn, rv_obj_rn = np.min, np.max
        elif opti_obj == 'max':
            obj_fn, rv_obj_rn = np.max, np.min
        else:
            raise TypeError('Please use "min" or "max" for opti_obj.')

        start_end_step_array = []
        for space in search_space_argvs:
            start, end = space
            step = (end - start) / search_resolution
            start_end_step_array.append([start, end, step])

        results = for_looper(score_function, loops_start_end_step=start_end_step_array,
                             tensorify_result=False)
        scores = [x[1]['score'] for x in results]
        best_score = obj_fn(scores)
        worst_score = rv_obj_rn(scores)

        best_pos = Hidden.parse_numpy_where_results(np.where(scores == best_score))[0]
        best_intervals = results[best_pos][0]

        if np.abs(best_score - worst_score) <= convergence_limit or max_iteration == 0:
            if max_iteration == 0:
                print('Max iter reached.')
            else:
                print('Results converged.')
            output = results[best_pos][1]
            output['best interval'] = best_intervals
            return results[best_pos][1]

        new_steps = [x[2] / 2 for x in start_end_step_array]

        new_max_space = best_intervals + new_steps
        new_min_space = best_intervals - new_steps

        if not search_beyond_original_space:
            new_max_space = [new_max_space[i] if new_max_space[i] < x[1] else x[1] for i, x in
                             enumerate(start_end_step_array)]
            new_min_space = [new_min_space[i] if new_min_space[i] > x[0] else x[0] for i, x in
                             enumerate(start_end_step_array)]

        new_search_space = [[new_min_space[i], new_max_space[i]] for i, x in enumerate(new_max_space)]

        return Optimisers.n_ary_search_optimization(score_function=score_function, search_space_argvs=new_search_space,
                                                    search_resolution=search_resolution,
                                                    max_iteration=max_iteration - 1,
                                                    convergence_limit=convergence_limit, opti_obj=opti_obj,
                                                    search_beyond_original_space=search_beyond_original_space)


# tests
class Tests:
    @staticmethod
    def test_for_looper():
        def loop_test(x):
            output = ''
            for i in x:
                output += str(i) + ','
            print(output)
            return np.random.rand()

        result = for_looper(loop_test, [[1, 5, 0.7], [20, 30, 2], [100, 200, 25.6]], tensorify_result=True)
        return result

    @staticmethod
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
        def reverse_dist_looper_wrapper(input):
            size = input[0]
            imba_factor = input[1]
            new_x = transform_1D_data_to_reverse_dist(x, no_of_new_samples=size,
                                                      return_same_sized_combined_dist=True,
                                                      imba_f=imba_factor, show_visualization=False)
            loss_b = helper(np.asarray(new_x))
            loss_std = np.max(loss_b)
            return {'score': loss_std, 'loss': loss_b}

        scores_b = Optimisers.n_ary_search_optimization(score_function=reverse_dist_looper_wrapper,
                                                        convergence_limit=0.05,
                                                        search_space_argvs=[[0, 2], [0, 2]], search_resolution=4)
        loss_b = scores_b['loss']

        # loss with optimized 'DOPE' sampling
        def DOPE_sampling_wrapper(input):
            new_sample_ratio = input[0]
            std = input[1]
            new_x_doped = transform_1D_samples_using_DOPE(x, return_same_sized_combined_dist=True,
                                                          new_sample_ratio=new_sample_ratio,
                                                          no_of_std=std)
            loss_c = helper(np.asarray(new_x_doped))
            loss_std = np.max(loss_c)
            return {'score': loss_std, 'loss': loss_c}

        scores_c = Optimisers.n_ary_search_optimization(score_function=DOPE_sampling_wrapper,
                                                        convergence_limit=0.05,
                                                        search_space_argvs=[[0, 2], [0, 2]], search_resolution=4)
        loss_c = scores_c['loss']

        x = np.arange(len(loss_a))
        plt.figure(2)
        plt.plot(x, loss_a, 'r--', x, loss_b, 'b--', x, loss_c, 'g--')
        plt.show()
        np.savetxt('losses2.csv', np.asarray([loss_a, loss_b, loss_c]), delimiter=',')

        return


if __name__ == '__main__':
    Tests.test_x()
