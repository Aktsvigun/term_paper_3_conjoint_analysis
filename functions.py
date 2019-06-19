import numpy as np
import pandas as pd
import pyreadstat
import pymc3 as pm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Process, current_process, Manager 

def plot_traces(traces, retain=1000):
    """
    Convenience function: plot traces with overlaid means and values
    """

    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')
        
def get_lr_estimates(df, target, random_state=0):
    """
    
        Get weights estimates using logistic regression.
    
        arguments:
            df: np.array, matrix of features.
            target: list or np.array, vector of target values.
            
        returns:
            estimates for weights using logistic regression.
    """
    lr = LogisticRegression(random_state=random_state)
    lr.fit(df, target)
    return np.concatenate((lr.intercept_, lr.coef_[0]))

def diff_model(df, features, random_state=0):
    
    """
    
        Builds a DataFrame with each row as a difference of the winning alternative and another one or vice versa.
        
        arguments:
            df: pd.DataFrame, which to parse.
            features: list or np.array or tuple, characteristics of each alternative.
            random_state: int, default=0, random state for numpy.random.seed.
            
        returns:
            df with difference as each row. Length == num_people * num_cards * (num_alternatives - 1)
            
    """
    df_diff = pd.DataFrame(columns=np.append(features, 'target'))
    # Fix random seed
    np.random.seed(random_state)
    # For each person
    for num_person in range(len(df)):
        # For each card
        for num_test in range(1, 8):
            # Winning alternative number
            win_number = df.loc[num_person, f'T{num_test}_select']
            # Winning card's characteristics
            char_win = df.loc[num_person, [f'T{num_test}_C{win_number}_{features[i]}' \
                                               for i in range(10)]].values
            # For each of lose alternatives
            for num_obj in np.delete(range(1, 6), win_number-1):
                char_lose = df.loc[num_person, [f'T{num_test}_C{num_obj}_{features[i]}'\
                                                for i in range(10)]].values
                # What to add to target, 0 or 1
                win_first = np.random.randint(2)
                if win_first:
                    df_diff.loc[len(df_diff)] = np.append(char_win - char_lose, win_first)
                else:
                    df_diff.loc[len(df_diff)] = np.append(char_lose - char_win, win_first)
    return df_diff

def personal_model(df, features, model='lr', priors=None, df_slice=None, estimates=None, random_state=0):
    
    """
    
        Builds a DataFrame with each row as a difference of the winning alternative and another one or vice versa
        
        arguments:
            df: pd.DataFrame, which to parse.
            features: list or np.array, characteristics of each alternative.
            model: 'lr' or 'bayes', default='lr', which model to use.
            priors: dict, default=None, dict of prior values for estimates if model='bayes'.
            df_slice: tuple or int, default=None, range(df_slice) is the slice of the df over which to iterate.
            estimates: list, default=None, a list if needed to append the results.
            random_state: int, default=0, random state for numpy.random.seed.

        returns:
            df with difference as each row. Length == num_people * num_cards * (num_alternatives - 1)
            
    """
    print(f'Process {current_process().name} started!')
    np.random.seed(random_state)
    if estimates == None:
        estimates = []
    if not df_slice:
        df_slice = (len(df))
    for num_person in range(*df_slice):
        df_person = pd.DataFrame(columns=np.append(features, 'target'))
        for num_test in range(1, 8):
            # For each object except for the winning:
            win_number = df.loc[num_person, f'T{num_test}_select']
            char_win = df.loc[num_person, [f'T{num_test}_C{win_number}_{features[i]}' \
                                               for i in range(10)]].values
            for num_obj in np.delete(range(1, 6), win_number-1):
                char_lose = df.loc[num_person, [f'T{num_test}_C{num_obj}_{features[i]}' \
                                                for i in range(10)]].values
                win_first = np.random.randint(2)
                if win_first:
                    df_person.loc[len(df_person)] = np.append(char_win - char_lose, win_first)
                else:
                    df_person.loc[len(df_person)] = np.append(char_lose - char_win, win_first)
        df_person[['Payment', 'Personalization', 'Price']] *= -1
        
        if model=='bayes':
            with pm.Model() as logistic_model: 
                
                pm.glm.GLM.from_formula('target ~ {0}'.format(' '.join(list(map(lambda x: str(x)+' '+'+',
                df_person.columns[:-1])))[:-2]), data=df_person, family=pm.glm.families.Binomial(), priors=priors)
                trace_logistic_model = pm.sample(2000, step=pm.NUTS(), chains=1, tune=1000)
                estimates.append(np.mean(list(map(lambda x: list(x.values()), trace_logistic_model)), axis=0))
                
        else:
            estimates.append(get_lr_estimates(df_person.values[:, :-1], df_person.target.values))
            
    print(f'Process {current_process().name} finished!')
    return np.array(estimates)

def visualize_estimates(estimates, annotate=True, use_features=None, three_dim=True,
                        use_tsne=True, perplexity=6, features=None, enlight=None, savefig=False):
    
    """
    
        Visualizes the vectors of any dimensionality in 2-d or 3-d using t-SNE if needed
        
        arguments:
            estimates: list or np.array, what to visualize.
            annotate: boolean, default=True, whether to annotate the points on the graph.
            use_features: list or np.array or list of str or None, default=None, which features to visualize.
            If array of ints, responding columns from estimates are selected. If list of str, responding
            values for features are selected from estimates. If None, all the estimates array is selected.
            three_dim: boolean, default=True, whether to visualize in 3-d when using t-SNE.
            use_tsne: boolean, default=True, whether to use t-SNE algorithm to visualize. ¡Important! 
            if False, features dimensionality must be equal to 2 or 3, either ValueError.
            perplexity: int, default=6, which perplexity to use if t-SNE is chosen.
            features: list or np.array, default=None, which features to visualize if use_features consists of str.
            enlight: int or None, default=None, the number of feature (starting with 0) by which to enlight the points.
            savefig: boolean, default=False, whether to save graph or not.
            
        returns:
            None; depicts the graph/s into the environment.
            
    """
    if isinstance(enlight, int):
        color = estimates[:, enlight]
    else:
        color = 'Aquamarine'
    if not use_features:
        tsne = TSNE(2, perplexity)
        points = tsne.fit_transform(estimates)
        if annotate:
            plt.figure(figsize=(25, 25))
            ax = plt.gca()
            ax.set_facecolor('AliceBlue')
            plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
            for n in range(len(points)):
                plt.annotate(f'{n}', xy=points[n, :], textcoords='data')
        else:
            plt.figure(figsize=(10, 5))
            ax = plt.gca()
            ax.set_facecolor('AliceBlue')
            plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
         
        if type(enlight) == int:
            plt.colorbar()
        if savefig:
            plt.savefig('estimates_2_dim.png')
            
        if three_dim:
            tsne = TSNE(3, perplexity)
            points = tsne.fit_transform(estimates)
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2]);
            if savefig:
                plt.savefig('estimates_3_dim.png')
    else:
        if type(use_features[0])=='str':
            use_features = [np.where(features==i)[0][0] for i in use_features]
        if use_tsne:
            tsne = TSNE(2, perplexity)
            points = tsne.fit_transform(estimates[:, use_features])
            if annotate:
                plt.figure(figsize=(25, 25))
                ax = plt.gca()
                ax.set_facecolor('AliceBlue')
                plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
                for n in range(len(points)):
                    plt.annotate(f'{n}', xy=points[n, :], textcoords='data')
            else:
                plt.figure(figsize=(10, 5))
                ax = plt.gca()
                ax.set_facecolor('AliceBlue')
                plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
            
            if type(enlight) == int:
                cbar=plt.colorbar()
                cbar.ax.set_ylabel(f'Values of feature {features[enlight-1]}', {'fontsize': 20}, labelpad=27, rotation=270)
            if savefig:
                plt.savefig('estimates_2_dim.png')
                
            if three_dim:
                tsne = TSNE(3, perplexity)
                points = tsne.fit_transform(estimates[:, use_features])
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points[:, 0], points[:, 1], points[:, 2]);
                if savefig:
                    plt.savefig('estimates_3_dim.png')
        else:
            if len(use_features)==2:
                points = estimates[:, use_features]
                if annotate:
                    plt.figure(figsize=(25, 25))
                    ax = plt.gca()
                    ax.set_facecolor('AliceBlue')
                    plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
                    for n in range(len(points)):
                        plt.annotate(f'{n}', xy=points[n, :], textcoords='data')
                else:
                    plt.figure(figsize=(10, 5))
                    ax = plt.gca()
                    ax.set_facecolor('AliceBlue')
                    plt.scatter(points[:, 0], points[:, 1], c=color, cmap='viridis', s=100)
                    
                if type(enlight) == int:
                    plt.colorbar()
                if savefig:
                    plt.savefig('estimates_2_dim.png')
                    
            elif len(use_features)==3:
                points = estimates[:, use_features]
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points[:, 0], points[:, 1], points[:, 2]);
                if savefig:
                    plt.savefig('estimates_3_dim.png')
            else:
                raise ValueError('Please, set 2 or 3 features to visualize or use t-SNE.')
                
def parallel_function(function, args, use_len=True):
    manager = Manager()
    procs = []
    res_1 = manager.list()
    res_2 = manager.list()
    res_3 = manager.list()
    res_4 = manager.list()
    res_5 = manager.list()
    res_6 = manager.list()
    res_7 = manager.list()
    res_8 = manager.list()
    
    if use_len:
        main_args = args[:-1]
        length = args[-1]
        
        proc_1 = Process(target=function, args=np.append(main_args, [(0, length//8), res_1]))
        procs.append(proc_1)

        proc_2 = Process(target=function, args=np.append(main_args, [(length//8, length*2//8), res_2]))
        procs.append(proc_2)

        proc_3 = Process(target=function, args=np.append(main_args, [(length*2//8, length*3//8), res_3]))
        procs.append(proc_3)

        proc_4 = Process(target=function, args=np.append(main_args, [(length*3//8, length*4//8), res_4]))
        procs.append(proc_4)

        proc_5 = Process(target=function, args=np.append(main_args, [(length*4//8, length*5//8), res_5]))
        procs.append(proc_5)

        proc_6 = Process(target=function, args=np.append(main_args, [(length*5//8, length*6//8), res_6]))
        procs.append(proc_6)

        proc_7 = Process(target=function, args=np.append(main_args, [(length*6//8, length*7//8), res_7]))
        procs.append(proc_7)

        proc_8 = Process(target=function, args=np.append(main_args, [(length*7//8, length), res_8]))
        procs.append(proc_8)

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    estimates = np.concatenate((res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8))
    return estimates

if __name__ == '__main__':
    df, meta = pyreadstat.read_sav('conjoint_host_sim_dummy.sav')
    for n in range(1, 8):
        df[f'T{n}_select'] = df[f'T{n}_select'].astype(int)
    features = np.delete(np.unique(list(map(lambda x: x[x.rindex('_')+1:],
                                            df.columns[2:]))), -1)
    df_diff = diff_model(df, features)
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('target ~ {0}'.format(' '.join(
            list(map(lambda x: str(x)+' '+'+', df_diff.columns[:-1])))[:-2]),
            data=df_diff, family=pm.glm.families.Binomial())
        trace_logistic_model = pm.sample(2000, step=pm.NUTS(),
                                         chains=1, tune=1000)
    plot_traces(trace_logistic_model);
    print(pm.summary(trace_logistic_model))
    
    priors = dict()
    priors['Intercept'] = pm.Laplace.dist(0, 0.2)
    priors['Gigabytes'] = pm.HalfStudentT.dist(20, 0.18)
    priors['Hostprovider1'] = pm.HalfStudentT.dist(20, 0.8)
    priors['Hostprovider2'] = pm.Lognormal.dist(-1.3, 1)
    priors['Hostprovider3'] = pm.Lognormal.dist(-2, 1)
    priors['Minutes'] = pm.Lognormal.dist(-1.3, 1.5)
    priors['Payment'] = pm.Lognormal.dist(-3.2, 1)
    priors['Personalization'] = pm.Lognormal.dist(-2.5, 1)
    priors['Price'] = pm.Wald.dist(0.5, 0.5)
    priors['Quantitysim2'] = pm.HalfStudentT.dist(20, 0.2)
    priors['Quantitysim3'] = pm.Wald.dist(0.2, 0.5)
    
    names_estimates = parallel_function(personal_model,
                                        (df, features, 'bayes',
                                         priors, len(df)))
    estimates = np.hstack((np.expand_dims(names_estimates[:, 0], -1),
                           names_estimates[:, 11:]))
    visualize_estimates(estimates, savefig=True, features=features,
                        enlight=1, three_dim=False)