import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from copy import deepcopy
import numpy as np 
from matplotlib.ticker import MaxNLocator
import os 


def create_boxplot_merged_experiments(df, column_name, epsilon, fig_folder_name, colors, title ="", file_name=""): 
    sns.set_theme(style='darkgrid')
    
    sns.boxplot(data=df.query('rmse == @column_name & epsilon == @epsilon'), x='area_name', y='value', hue='postprocessing', palette=colors, width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=2, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Geometric Mechanism with Epsilon = {epsilon} {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_{epsilon}{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_{epsilon}{file_name}.svg')), bbox_inches='tight')


def create_boxplot_experiments(df, column_name, fig_folder_name, colors, title = "", file_name=""): 
    sns.set_theme(style='darkgrid')
    
    sns.boxplot(data=df.query('rmse == @column_name'), x='area_name', y='value', hue='epsilon', palette=colors, width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=4, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Geometric Mechanism Overview {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_epsilons{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_epsilons{file_name}.svg')), bbox_inches='tight')


def create_boxplot_merged_experiments_laplace(df, column_name, epsilon, fig_folder_name, colors, title ="", file_name="", ncol=2): 
    sns.set_theme(style='darkgrid')
    
    sns.boxplot(data=df.query('rmse == @column_name & epsilon == @epsilon'), x='area_name', y='value', hue='postprocessing', palette=colors, width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=ncol, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Laplace Mechanism with Epsilon = {epsilon} {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_laplace_{epsilon}{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_laplace_{epsilon}{file_name}.svg')), bbox_inches='tight')


def create_boxplot_experiments_laplace(df, column_name, fig_folder_name, colors, title = "", file_name=""): 
    sns.set_theme(style='darkgrid')
    
    sns.boxplot(data=df.query('rmse == @column_name'), x='area_name', y='value', hue='epsilon', palette=colors, width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=4, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Laplace Mechanism Overview {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_laplace_epsilons{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_laplace_epsilons{file_name}.svg')), bbox_inches='tight')


def create_boxplot_experiments_kl(df, column_name, mu,fig_folder_name, colors, title = "", file_name=""): 
    sns.set_theme(style='darkgrid')
    
    sns.boxplot(data=df.query('kl_divergence == @column_name & mu == @mu'), x='area_name', y='value', hue='epsilon', palette=colors, width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=4, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('KL Divergence')
    plt.title(f'KL Divergence Overview with mu={mu} {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_epsilons_{mu}{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_epsilons_{mu}{file_name}.svg')), bbox_inches='tight')


def create_boxplot_merged_experiments_kl(df, column_name, epsilon, fig_folder_name, title ="", file_name=""): 
    sns.set_theme(style='darkgrid')
    colors = ['red', 'blue', 'orange', 'green', 'purple', 'yellow']
    sns.boxplot(data=df.query('kl_divergence == @column_name & epsilon == @epsilon'), palette=colors, x='area_name', y='value', hue='mu', width=0.5, gap=0.2)
    plt.legend(bbox_to_anchor=(0.5, -0.22), loc='lower center', ncol=6, borderaxespad=0, fontsize=10)
    plt.xlabel('Wards')
    plt.ylabel('KL Divergence')
    plt.title(f'KL Divergence with Epsilon = {epsilon} {title}')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_{epsilon}{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_{epsilon}{file_name}.svg')), bbox_inches='tight')


def kl_experiments_scatter(df, mu, ward_labels, color, fig_folder_name, title="", filename=""):

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(14,8))
    p = sns.relplot(x='epsilon', y='value', col='area_name', hue='area_name', palette=color, 
                    data=df.query('kl_divergence == "kl_divergence" & mu == @mu'), col_wrap=3, height=3, s=20)
    plt.suptitle(f'KL Divergence for mu={mu} {title}', fontsize=14)
    sns.move_legend(p, "upper right", bbox_to_anchor=(0.99, 0.9), ncol=1, title='', frameon=False)

    for i, ax in enumerate(p.axes.flatten()):
        ax.set_title(ward_labels[i])
        ax.tick_params(labelbottom=True)
        ax.set_ylabel('KL Divergence', fontsize=10)
        ax.set_xlabel('Epsilon', visible=True, fontsize=10)

    plt.subplots_adjust(top=0.9, wspace=None, hspace=0.35)

    plt.savefig((os.path.join(fig_folder_name, f'kl_areas_mu_0.0001_experiments{filename}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'kl_areas_mu_0.0001_experiments{filename}.svg')), bbox_inches='tight')
    

def kl_experiments_line(df, mu, ward_labels, color, fig_folder_name, title="", filename=""):

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(14,8))
    p = sns.relplot(x='epsilon', y='value', col='area_name', hue='area_name', palette=color, 
                    data=df.query('kl_divergence == "kl_divergence" & mu == @mu'), col_wrap=3, height=3, 
                    kind='line', marker='o', markersize=5, markeredgecolor=None, estimator=np.mean)


    plt.suptitle(f'KL Divergence for mu={mu} {title}', fontsize=14)
    sns.move_legend(p, "upper right", bbox_to_anchor=(0.99, 0.9), ncol=1, title='', frameon=False)

    for i, ax in enumerate(p.axes.flatten()):
        ax.set_title(ward_labels[i])
        ax.tick_params(labelbottom=True)
        ax.set_ylabel('KL Divergence', fontsize=10)
        ax.set_xlabel('Epsilon', visible=True, fontsize=10)

    plt.subplots_adjust(top=0.9, wspace=None, hspace=0.35)

    plt.savefig((os.path.join(fig_folder_name, f'kl_areas_mu_0.0001_experiments_line{filename}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'kl_areas_mu_0.0001_experiments_line{filename}.svg')), bbox_inches='tight')



def rmse_experiments(df, ward_labels, color, fig_folder_name, title="", filename=""):

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(14,8))
    p = sns.relplot(x='epsilon', y='value', col='area_name', hue='area_name', palette=color, data=df.query('rmse == "rmse_dp"'), col_wrap=3, height=3, 
                    kind='line', marker='o', markersize=5, markeredgecolor=None, estimator=np.mean)


    plt.suptitle(f'RMSE {title}', fontsize=14)
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.0, 0.9), ncol=1, title='', frameon=False)

    for i, ax in enumerate(p.axes.flatten()):
        ax.set_title(ward_labels[i])
        ax.tick_params(labelbottom=True)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_xlabel('Epsilon', visible=True, fontsize=10)

    plt.subplots_adjust(top=0.9, wspace=None, hspace=0.35)

    plt.savefig((os.path.join(fig_folder_name, f'rmse_experiments_line{filename}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'rmse_experiments_line{filename}.svg')), bbox_inches='tight')


def rmse_experiments_scatter(df, ward_labels, color, fig_folder_name, title="", filename=""):

    plt.style.use('seaborn-v0_8-dark')
    plt.figure(figsize=(14,8))
    p = sns.relplot(x='epsilon', y='value', col='area_name', hue='area_name', palette=color, data=df.query('rmse == "rmse_dp"'), col_wrap=3, height=3)


    plt.suptitle(f'RMSE {title}', fontsize=14)
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.0, 0.9), ncol=1, title='', frameon=False)

    for i, ax in enumerate(p.axes.flatten()):
        ax.set_title(ward_labels[i])
        ax.tick_params(labelbottom=True)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_xlabel('Epsilon', visible=True, fontsize=10)

    plt.subplots_adjust(top=0.9, wspace=None, hspace=0.35)

    plt.savefig((os.path.join(fig_folder_name, f'rmse_experiments_scatter{filename}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'rmse_experiments_scatter{filename}.svg')), bbox_inches='tight')



def create_table_df_dp(table_features, df, epsilons):
    reduced_df = df.loc[epsilons].copy(deep=True)

    reduced_df.reset_index(inplace=True)
    index= pd.MultiIndex.from_frame(reduced_df[['epsilon', 'area_name']])
    reduced_df.set_index(index, inplace=True)
    reduced_df.drop(['epsilon', 'area_name'], axis=1, inplace=True)

    return reduced_df[table_features]


def create_table_dp(table_features, df_clipping, df, epsilons):
    table_no_clip = create_table_df_dp(table_features, df, epsilons)
    table_clip = create_table_df_dp(table_features, df_clipping, epsilons)

    dp_df = pd.concat([table_clip, table_no_clip], axis=1)

    latex_table = dp_df.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)


def create_table_df_dp_data_error(table_features, table_features_data_error, df, epsilons):
    reduced_df = df.loc[epsilons].copy(deep=True)

    reduced_df.reset_index(inplace=True)
    index= pd.MultiIndex.from_frame(reduced_df[['epsilon', 'area_name']])
    reduced_df.set_index(index, inplace=True)
    reduced_df.drop(['epsilon', 'area_name'], axis=1, inplace=True)

    data_error_df = reduced_df[table_features_data_error].loc[epsilons[0]].copy()
    data_error_df.reset_index(inplace=True)
    data_error_df['epsilon'] = 0 

    index_2 = pd.MultiIndex.from_frame(data_error_df[['epsilon', 'area_name']])
    data_error_df.set_index(index_2, inplace=True)
    data_error_df.drop(['epsilon', 'area_name'], axis=1, inplace=True)
    data_error_df.columns= table_features

    data_error_dp_df = pd.concat([data_error_df, reduced_df[table_features]])

    return data_error_dp_df


def create_table_dp_data_error(table_features, table_features_data_error, df_clipping, df, epsilons):
    table_no_clip = create_table_df_dp_data_error(table_features, table_features_data_error, df, epsilons)
    table_clip = create_table_df_dp_data_error(table_features, table_features_data_error, df_clipping, epsilons)

    data_error_dp_df = pd.concat([table_clip, table_no_clip], axis=1)

    latex_table = data_error_dp_df.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)


def create_pop_df(table_features, table_features_data_error, df, epsilons):
    reduced_df = df.loc[epsilons].copy(deep=True)

    reduced_df.reset_index(inplace=True)
    index= pd.MultiIndex.from_frame(reduced_df[['epsilon', 'area_name']])
    reduced_df.set_index(index, inplace=True)
    reduced_df.drop(['epsilon', 'area_name'], axis=1, inplace=True)

    data_error_df = reduced_df[table_features_data_error].loc[epsilons[0]].copy()
    data_error_df.reset_index(inplace=True)
    data_error_df['epsilon'] = 0 

    index_2 = pd.MultiIndex.from_frame(data_error_df[['epsilon', 'area_name']])
    data_error_df.set_index(index_2, inplace=True)
    data_error_df.drop(['epsilon', 'area_name'], axis=1, inplace=True)
    data_error_df.columns= table_features

    data_error_dp_df = pd.concat([data_error_df, reduced_df[table_features]])

    return data_error_dp_df


def create_table_pop(table_features, table_features_dp, table_features_data_error, table_features_data_error_dp, df_clipping, df, epsilons):
    table_dp_clip = create_table_df_dp_data_error(table_features_dp, table_features, df_clipping, epsilons)
    table_dp = create_table_df_dp_data_error(table_features_dp, table_features, df, epsilons)

    table_data_error_dp_clip = create_table_df_dp_data_error(table_features_data_error_dp, table_features_data_error, df_clipping, epsilons)
    table_data_error_dp = create_table_df_dp_data_error(table_features_data_error_dp, table_features_data_error, df, epsilons)

    table_df = pd.concat([table_dp_clip, table_data_error_dp_clip, table_dp, table_data_error_dp], axis=1)

    latex_table = table_df.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)


def create_table_pop_2(table_features, table_features_dp, table_features_data_error, table_features_data_error_dp, df_clipping, df, epsilons):
    table_ground_truth = create_table_df_dp(table_features, df, epsilons)
    table_data_error = create_table_df_dp(table_features_data_error, df, epsilons)
    
    table_dp_clip = create_table_df_dp(table_features_dp, df_clipping, epsilons)
    table_dp = create_table_df_dp(table_features_dp, df, epsilons)

    table_data_error_dp_clip = create_table_df_dp(table_features_data_error_dp, df_clipping, epsilons)
    table_data_error_dp = create_table_df_dp(table_features_data_error_dp, df, epsilons)

    table_df = pd.concat([table_ground_truth, table_data_error, table_dp_clip, table_data_error_dp_clip, table_dp, table_data_error_dp], axis=1)

    latex_table = table_df.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)

def plot_kl_divergence(colors, df_kl, ward_codes, labels_wards, fig_folder_name, fig_name):
    plt.style.use('seaborn-v0_8-dark')
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)

    for i, code in enumerate(ward_codes):

        area_geo_clip = df_kl.query('area_code == @code').copy(deep=True)
        df = area_geo_clip[area_geo_clip.index.isin([0.001], level=1)]
        df.reset_index(inplace=True)

        axs[i].plot(df['epsilon'], df['kl_divergence'], color=colors[1], label='KL Divergence DP')
        axs[i].plot(df['epsilon'], df['kl_divergence_data_error'], color=colors[0], label='KL Divergence Data Error')
        axs[i].plot(df['epsilon'], df['kl_divergence_data_error_dp'], color=colors[2], label='KL Divergence Data Error and DP')

        axs[i].set_title(labels_wards[i], fontsize=11)
        axs[0].set_ylabel('KL Divergence', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        fig.subplots_adjust(wspace=0.1)

    fig.suptitle(f'KL Divergence for mu = 0.001', fontsize=15, y=1.02)

    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.savefig((os.path.join(fig_folder_name, f'kl_divergence_compare_{fig_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'kl_divergence_compare_{fig_name}.svg')), bbox_inches='tight')


def plot_rmse(colors, df_geo, df_geo_clip, df_laplace, df_laplace_clip, df_laplace_round_clip, ward_codes, labels_wards, fig_folder_name):

    plt.style.use('seaborn-v0_8-dark')
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True)
    plt.yscale('symlog')

    for i, code in enumerate(ward_codes):

        area_geo = df_geo.query('area_code == @code').copy(deep=True)
        area_geo_clip = df_geo_clip.query('area_code == @code').copy(deep=True)

        area_laplace = df_laplace.query('area_code == @code')
        area_laplace_clip = df_laplace_clip.query('area_code == @code')
        area_laplace_round_clip = df_laplace_round_clip.query('area_code == @code')

        area_geo.reset_index(inplace=True)
        area_geo_clip.reset_index(inplace=True)

        area_laplace.reset_index(inplace=True)
        area_laplace_clip.reset_index(inplace=True)
        area_laplace_round_clip.reset_index(inplace=True)

        axs[0][i].plot(area_geo['epsilon'], area_geo['rmse_dp'], color=colors[1], label='Geometric without post-processing')
        axs[0][i].plot(area_geo_clip['epsilon'], area_geo_clip['rmse_dp'], color=colors[2], label='Geometric with clipping')
        axs[0][i].plot(area_geo['epsilon'], area_geo['rmse_data_error'], color=colors[0], label='Data Error', linestyle= '--')

        axs[0][2].legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, fontsize=10)

        axs[1][i].plot(area_laplace['epsilon'], area_laplace['rmse_dp'], color=colors[1], label='Laplace without post-processing')
        axs[1][i].plot(area_laplace_clip['epsilon'], area_laplace_clip['rmse_dp'], color=colors[2], label='Laplace with clipping')
        axs[1][i].plot(area_laplace_round_clip['epsilon'], area_laplace_round_clip['rmse_dp'], color=colors[3], label='Laplace with clipping and rounding')
        # using the data error from the geometric table for better comparison
        axs[1][i].plot(area_geo['epsilon'], area_geo['rmse_data_error'], color=colors[0], label='Data Error', linestyle= '--')

        axs[1][2].legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0, fontsize=10)

        axs[0][i].set_title(labels_wards[i], fontsize=11)
        axs[0][0].set_ylabel('RMSE', fontsize=10)
        axs[1][i].set_xlabel('Epsilon', fontsize=10)

        fig.subplots_adjust(wspace=0.1)

    fig.suptitle(f'RMSE for Geometric and Laplace Mechanism', fontsize=15, y=0.98)
    plt.savefig((os.path.join(fig_folder_name, 'rmse_compare.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, 'rmse_compare.svg')), bbox_inches='tight')
    

def create_bar_plot_pop(wards, ethnicities, epsilons, colors, step, width, gap, fig_name, fig_folder_name):
    plt.style.use('seaborn-v0_8-dark')

    y = np.arange(stop=((len(ethnicities)*step)-1), step=step)
    for epsilon in epsilons: 

        for i, ward in wards.items():

            plt.style.use('seaborn-v0_8-dark') 
            plt.figure(figsize=(8, 5))

            plt.rc('font', size=12)  # controls default text sizes
            plt.rc('axes', titlesize=14)  # fontsize of the axes title
            plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=11)  # fontsize of the tick labels 

            df = ward[0]
    
            labels = ['Ground Truth', f'+ DP Noise with Epsilon = {epsilon}', '+ Data Error']
            shift = width+gap

            plt.barh(y-shift, df[f'PopulationNumbersDP {epsilon}'], width, color=colors[1], label=labels[1])
            plt.barh(y, df['PopulationNumbers'], width, color=colors[0], label=labels[0])
            plt.barh(y+shift, df[f'PopulationNumbersDataError'], width, color=colors[2], label=labels[2])
     
            yticks = [i+(width/2) for i in y]
            plt.yticks(yticks, ethnicities)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            plt.ylabel('Ethnic Group')
            plt.xlabel('Population Numbers')

            plt.margins(y=0)

            plt.title(f'Ethnicities in {ward[1]['area_name']}')
            plt.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.png')), bbox_inches='tight')
            plt.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.svg')), bbox_inches='tight')


def create_bar_plot_pop_dp_on_data_error(wards, ethnicities, epsilons, colors, step, width, gap, fig_name, fig_folder_name):
    plt.style.use('seaborn-v0_8-dark')

    y = np.arange(stop=((len(ethnicities)*step)-1), step=step)
    for epsilon in epsilons: 

        for i, ward in wards.items():

            plt.style.use('seaborn-v0_8-bright') 
            plt.figure(figsize=(8, 5))

            plt.rc('font', size=12)  # controls default text sizes
            plt.rc('axes', titlesize=14)  # fontsize of the axes title
            plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=11)  # fontsize of the tick labels 

            df = ward[0]

            labels = ['Ground Truth', f'+ DP Noise with Epsilon = {epsilon}', '+ Data Error', f'+ Data Error + DP Noise with Epsilon = {epsilon}']
            shift = width+gap

            plt.barh(y+(2*shift), df['PopulationNumbers'], width, color=colors[0], label=labels[0])
            plt.barh(y+shift, df[f'PopulationNumbersDataError'], width, color=colors[1], label=labels[2])
            plt.barh(y, df[f'PopulationNumbersDP {epsilon}'], width, color=colors[2], label=labels[1])
            plt.barh(y-shift, df[f'PopulationNumbersDataErrorDP {epsilon}'], width, color=colors[3], label=labels[3])
     
            yticks = [i+width for i in y]

            plt.yticks(yticks, ethnicities)

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            plt.ylabel('Ethnic Group')
            plt.xlabel('Population Numbers')

            plt.margins(y=0)

            plt.title(f'Ethnicities in {ward[1]['area_name']}')
            plt.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.png')), bbox_inches='tight')
            plt.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.svg')), bbox_inches='tight')


def create_several_bar_plot_pop_dp_on_data_error(wards, ethnicities, epsilons, colors, step, width, gap, fig_name, fig_folder_name):
    plt.style.use('seaborn-v0_8-dark')

    for i, ward in wards.items():
        fig, ax = plt.subplots(nrows=1, ncols=len(epsilons), figsize=(14,4), sharey=True)
        fig.suptitle(f'Ethnicities in {ward[1]['area_name']}', fontsize=14)

        shift = width+gap

        j=0

        y = np.arange(stop=((len(ethnicities)*step)-1), step=step)
        for epsilon in epsilons: 
    
            plt.rc('font', size=12)  # controls default text sizes
            plt.rc('axes', titlesize=14)  # fontsize of the axes title
            plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=11)  # fontsize of the tick labels 

            df = ward[0]

            labels = ['Ground Truth', f'+ DP Noise', '+ Data Error', f'+ Data Error + DP Noise']

            ax[j].barh(y+(2*shift), df['PopulationNumbers'], width, color=colors[0], label=labels[0])
            ax[j].barh(y+shift, df[f'PopulationNumbersDataError'], width, color=colors[1], label=labels[2])
            ax[j].barh(y, df[f'PopulationNumbersDP {epsilon}'], width, color=colors[2], label=labels[1])
            ax[j].barh(y-shift, df[f'PopulationNumbersDataErrorDP {epsilon}'], width, color=colors[3], label=labels[3])
 
            ax[0].set_ylabel('Ethnic Group', fontsize=10)
            ax[j].set_xlabel('Population Numbers', fontsize=10)
            ax[j].set_title(f'Epsilon = {epsilon}', fontsize=11)

            yticks = [i+width for i in y]
            ax[j].set_yticks(yticks, ethnicities)
            ax[j].margins(y=0)
        
            j += 1

        lines, labels = ax[0].get_legend_handles_labels() 
        fig.legend(lines, labels, bbox_to_anchor=(0.91, 0.84), loc='upper left', borderaxespad=0, prop={'size': 10}) 
        fig.subplots_adjust(top=0.86, wspace=0.1)
    
        fig.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.png')), bbox_inches='tight')
        fig.savefig((os.path.join(fig_folder_name, f'{fig_name}_{ward[1]['area_name']}_{epsilon}.svg')), bbox_inches='tight')


def barplot_inc(gap, width, epsilons, df, ward_codes, labels_wards, fig_folder_name):
    plt.style.use('seaborn-v0_8-dark')

    width = width
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    sns.set_palette(palette='Paired')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for i, code in enumerate(ward_codes):

        area_df = df.query('area_code == @code').reset_index()

        axs[i].bar(0, area_df['number_ethnicities'], width, label='Ethnic groups total area count')
        axs[i].bar(0, area_df['number_minorities'], width, label='Minority groups total area count')

        axs[i].bar(x, area_df['significantly_increased'], width, label='Significantly increased groups total')
        axs[i].bar(x, area_df['significantly_increased_minority'], width, label='Significantly increased minority groups')

        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Increased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))


    fig.suptitle(f'Significantly Increased Groups', fontsize=16, y=1.0)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.savefig((os.path.join(fig_folder_name, 'significantly_increased__compare_bars_cat.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, 'significantly_increased__compare_bars_cat.svg')), bbox_inches='tight')


def barplot_dec(gap, width, epsilons, df, ward_codes, labels_wards, fig_folder_name):
    plt.style.use('seaborn-v0_8-dark')

    width = width
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    sns.set_palette(palette='Paired')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for i, code in enumerate(ward_codes):

        area_df = df.query('area_code == @code').reset_index()

        axs[i].bar(0, area_df['number_ethnicities'], width, label='Ethnic groups total area count')
        axs[i].bar(0, area_df['number_minorities'], width, label='Minority groups total area count')

        axs[i].bar(x, area_df['significantly_decreased'], width, label='Significantly decreased groups total')
        axs[i].bar(x, area_df['significantly_decreased_minority'], width, label='Significantly decreased minority groups')

        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Decreased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f'Significantly Decreased Groups', fontsize=16, y=1.0)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.savefig((os.path.join(fig_folder_name, 'significantly_decreased_compare_bars_cat.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, 'significantly_decreased_compare_bars_cat.svg')), bbox_inches='tight')
   

def barplot_inc_mean(gap, width, epsilons, df, column_name, ward_names, labels_wards, fig_folder_name, fig_name="", title=""):
    
    plt.style.use('seaborn-v0_8-dark')

    width = width
    barwidth = width/2
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    palette = sns.color_palette('Paired')

    color_zero = '#006d2c'

    fig, axs = plt.subplots(1, len(ward_names), figsize=(len(ward_names)*4, 4), sharey=True)

    for i, name in enumerate(ward_names):

        axs[i].bar(0, df.query('rmse == "number_ethnicities" & area_name == @name')['value'].mean(), barwidth, label='Ethnic groups total - ground truth', color=palette[0], linewidth=0)
        axs[i].bar(0, df.query('rmse == "number_minorities" & area_name == @name')['value'].mean(), barwidth, label='Minority groups - ground truth', color=palette[1], linewidth=0)

        for j, epsilon in enumerate (epsilons):

            df_inc = df.query('rmse == @column_name & area_name == @name & epsilon == @epsilon')['value']
            df_inc_minority = df.query(f'rmse == "{column_name}_minority" & area_name == @name & epsilon == @epsilon')['value']
            df_inc_zero = df.query(f'rmse == "{column_name}_zero" & area_name == @name & epsilon == @epsilon')['value']

            if (j == 0):

                print(f'df_inc_mean: {df_inc.mean()}, df_min_mean: {df_inc_minority.mean()}, df_zero_mean: {df_inc_zero.mean()}')

                axs[i].bar(x[j]-(barwidth/2), df_inc.mean(), barwidth, label='Significantly increased groups total', color=palette[2], linewidth=0)
                if ((df_inc.mean() > 0) and (df_inc.mean() > df_inc_minority.mean())):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_inc.mean(), yerr=df_inc.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_inc_minority.mean(), barwidth, label='Significantly increased minority groups', color=palette[3], linewidth=0)
                if (df_inc_minority.mean() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_inc_minority.mean(), yerr=df_inc_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_inc_zero.mean(), barwidth, label='Significantly increased empty groups', color=color_zero, linewidth=0)
                if (df_inc_zero.mean() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_inc_zero.mean(), yerr=df_inc_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

            else:
                axs[i].bar(x[j]-(barwidth/2), df_inc.mean(), barwidth, label='_hidden', color=palette[2], linewidth=0)
                if ((df_inc.mean() > 0) and (df_inc.mean() > df_inc_minority.mean())):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_inc.mean(), yerr=df_inc.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_inc_minority.mean(), barwidth, label='_hidden', color=palette[3], linewidth=0)
                if (df_inc_minority.mean() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_inc_minority.mean(), yerr=df_inc_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_inc_zero.mean(), barwidth, label='_hidden', color=color_zero, linewidth=0)
                if (df_inc_zero.mean() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_inc_zero.mean(), yerr=df_inc_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
        
        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Increased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.margins(y=0, x=0)
    fig.subplots_adjust(top=0.86, wspace=0.1)
    fig.suptitle(f'Significantly Increased Groups {title}- Mean', fontsize=16, y=1.0)

    if (len(ward_names) <= 3):
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    else:
        plt.legend(bbox_to_anchor=(-1.8, -0.3), loc='lower center', ncol=3, borderaxespad=0, fontsize=10)

    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_mean{fig_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_mean{fig_name}.svg')), bbox_inches='tight')



def barplot_inc_median(gap, width, epsilons, df, column_name, ward_names, labels_wards, fig_folder_name, fig_name="", title=""):
    
    plt.style.use('seaborn-v0_8-dark')

    width = width
    barwidth = width/2
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    palette = sns.color_palette('Paired')

    color_zero = '#006d2c'

    fig, axs = plt.subplots(1, len(ward_names), figsize=(len(ward_names)*4, 4), sharey=True)

    for i, name in enumerate(ward_names):

        axs[i].bar(0, df.query('rmse == "number_ethnicities" & area_name == @name')['value'].mean(), barwidth, label='Ethnic groups total - ground truth', color=palette[0], linewidth=0)
        axs[i].bar(0, df.query('rmse == "number_minorities" & area_name == @name')['value'].mean(), barwidth, label='Minority groups - ground truth', color=palette[1], linewidth=0)

        for j, epsilon in enumerate (epsilons):

            df_inc = df.query('rmse == @column_name & area_name == @name & epsilon == @epsilon')['value']
            df_inc_minority = df.query(f'rmse == "{column_name}_minority" & area_name == @name & epsilon == @epsilon')['value']
            df_inc_zero = df.query(f'rmse == "{column_name}_zero" & area_name == @name & epsilon == @epsilon')['value']

            if (j == 0):

                print(f'df_inc_median: {df_inc.median()}, df_min_median: {df_inc_minority.median()}, df_zero_median: {df_inc_zero.median()}')

                axs[i].bar(x[j]-(barwidth/2), df_inc.median(), barwidth, label='Significantly increased groups total', color=palette[2], linewidth=0)
                if ((df_inc.median() > 0) and (df_inc.median() > (df_inc_minority.median()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_inc.median(), yerr=df_inc.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_inc_minority.median(), barwidth, label='Significantly increased minority groups', color=palette[3], linewidth=0)
                if (df_inc_minority.median() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_inc_minority.median(), yerr=df_inc_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_inc_zero.median(), barwidth, label='Significantly increased empty groups', color=color_zero, linewidth=0)
                if (df_inc_zero.median() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_inc_zero.median(), yerr=df_inc_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

            else:
                axs[i].bar(x[j]-(barwidth/2), df_inc.median(), barwidth, label='_hidden', color=palette[2], linewidth=0)
                if ((df_inc.median() > 0) and (df_inc.median() > (df_inc_minority.median()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_inc.median(), yerr=df_inc.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_inc_minority.median(), barwidth, label='_hidden', color=palette[3], linewidth=0)
                if (df_inc_minority.median() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_inc_minority.median(), yerr=df_inc_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_inc_zero.median(), barwidth, label='_hidden', color=color_zero, linewidth=0)
                if (df_inc_zero.median() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_inc_zero.median(), yerr=df_inc_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Increased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.margins(y=0, x=0)
    fig.subplots_adjust(top=0.86, wspace=0.1)
    fig.suptitle(f'Significantly Increased Groups {title}- Median', fontsize=16, y=1.0)

    if (len(ward_names) <= 3):
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    else:
        plt.legend(bbox_to_anchor=(-1.8, -0.3), loc='lower center', ncol=3, borderaxespad=0, fontsize=10)

    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_median{fig_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_median{fig_name}.svg')), bbox_inches='tight')


def barplot_dec_mean(gap, width, epsilons, df, column_name, ward_names, labels_wards, fig_folder_name, fig_name="", title=""):
    
    plt.style.use('seaborn-v0_8-dark')

    width = width
    barwidth = width/2
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    palette = sns.color_palette('Paired')
    color_zero = '#006d2c'

    fig, axs = plt.subplots(1, len(ward_names), figsize=(len(ward_names)*4, 4), sharey=True)

    for i, name in enumerate(ward_names):

        axs[i].bar(0, df.query('rmse == "number_ethnicities" & area_name == @name')['value'].mean(), barwidth, label='Ethnic groups total - ground truth', color=palette[0], linewidth=0)
        axs[i].bar(0, df.query('rmse == "number_minorities" & area_name == @name')['value'].mean(), barwidth, label='Minority groups - ground truth', color=palette[1], linewidth=0)

        for j, epsilon in enumerate (epsilons):

            df_dec = df.query('rmse == @column_name & area_name == @name & epsilon == @epsilon')['value']
            df_dec_minority = df.query(f'rmse == "{column_name}_minority" & area_name == @name & epsilon == @epsilon')['value']
            df_dec_zero = df.query(f'rmse == "{column_name}_zero" & area_name == @name & epsilon == @epsilon')['value']

            if (j == 0):

                print(f'df_dec_mean: {df_dec.mean()}, df_min_mean: {df_dec_minority.mean()}, df_zero_mean: {df_dec_zero.mean()}')

                axs[i].bar(x[j]-(barwidth/2), df_dec.mean(), barwidth, label='Significantly decreased groups total', color=palette[2], linewidth=0)
                if ((df_dec.mean() > 0) and (df_dec.mean() > (df_dec_minority.mean()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_dec.mean(), yerr=df_dec.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_dec_minority.mean(), barwidth, label='Significantly decreased minority groups', color=palette[3], linewidth=0)
                if (df_dec_minority.mean() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_dec_minority.mean(), yerr=df_dec_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_dec_zero.mean(), barwidth, label='Significantly decreased empty groups', color=color_zero, linewidth=0)
                if (df_dec_zero.mean() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_dec_zero.mean(), yerr=df_dec_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

            else:
                axs[i].bar(x[j]-(barwidth/2), df_dec.mean(), barwidth, label='_hidden', color=palette[2], linewidth=0)
                if ((df_dec.mean() > 0) and (df_dec.mean() > (df_dec_minority.mean()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_dec.mean(), yerr=df_dec.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_dec_minority.mean(), barwidth, label='_hidden', color=palette[3], linewidth=0)
                if (df_dec_minority.mean() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_dec_minority.mean(), yerr=df_dec_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_dec_zero.mean(), barwidth, label='_hidden', color=color_zero, linewidth=0)
                if (df_dec_zero.mean() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_dec_zero.mean(), yerr=df_dec_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
       
        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Decreased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.margins(y=0, x=0)
    fig.subplots_adjust(top=0.86, wspace=0.1)
    fig.suptitle(f'Significantly Decreased Groups {title}- Mean', fontsize=16, y=1.0)

    if (len(ward_names) <= 3):
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    else:
        plt.legend(bbox_to_anchor=(-1.8, -0.3), loc='lower center', ncol=3, borderaxespad=0, fontsize=10)

    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_mean{fig_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_mean{fig_name}.svg')), bbox_inches='tight')

def barplot_dec_median(gap, width, epsilons, df, column_name, ward_names, labels_wards, fig_folder_name, fig_name="", title=""):
    
    plt.style.use('seaborn-v0_8-dark')

    width = width
    barwidth = width/2
    gap = gap
    shift = width+gap
    x = np.linspace(start=shift, stop=len(epsilons)*shift, num=len(epsilons))

    palette = sns.color_palette('Paired')
    color_zero = '#006d2c'
 
    fig, axs = plt.subplots(1, len(ward_names), figsize=(len(ward_names)*4, 4), sharey=True)

    for i, name in enumerate(ward_names):

        axs[i].bar(0, df.query('rmse == "number_ethnicities" & area_name == @name')['value'].mean(), barwidth, label='Ethnic groups total - ground truth', color=palette[0], linewidth=0)
        axs[i].bar(0, df.query('rmse == "number_minorities" & area_name == @name')['value'].mean(), barwidth, label='Minority groups - ground truth', color=palette[1], linewidth=0)

        for j, epsilon in enumerate (epsilons):

            df_dec = df.query('rmse == @column_name & area_name == @name & epsilon == @epsilon')['value']
            df_dec_minority = df.query(f'rmse == "{column_name}_minority" & area_name == @name & epsilon == @epsilon')['value']
            df_dec_zero = df.query(f'rmse == "{column_name}_zero" & area_name == @name & epsilon == @epsilon')['value']

            if (j == 0):

                print(f'df_dec_median: {df_dec.median()}, df_min_median: {df_dec_minority.median()}, df_zero_median: {df_dec_zero.median()}')

                axs[i].bar(x[j]-(barwidth/2), df_dec.median(), barwidth, label='Significantly decreased groups total', color=palette[2], linewidth=0)
                if ((df_dec.median() > 0) and (df_dec.median() > (df_dec_minority.median()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_dec.median(), yerr=df_dec.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_dec_minority.median(), barwidth, label='Significantly decreased minority groups', color=palette[3], linewidth=0)
                if (df_dec_minority.median() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_dec_minority.median(), yerr=df_dec_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_dec_zero.median(), barwidth, label='Significantly decreased empty groups', color=color_zero, linewidth=0)
                if (df_dec_zero.median() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_dec_zero.median(), yerr=df_dec_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

            else:
                axs[i].bar(x[j]-(barwidth/2), df_dec.median(), barwidth, label='_hidden', color=palette[2], linewidth=0)
                if ((df_dec.median() > 0) and (df_dec.median() > (df_dec_minority.median()))):
                    axs[i].errorbar(x[j]-(barwidth/2)-0.05, df_dec.median(), yerr=df_dec.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
                
                axs[i].bar(x[j]-(barwidth/2), df_dec_minority.median(), barwidth, label='_hidden', color=palette[3], linewidth=0)
                if (df_dec_minority.median() > 0):
                    axs[i].errorbar(x[j]-(barwidth/2)+0.05, df_dec_minority.median(), yerr=df_dec_minority.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)

                axs[i].bar(x[j]+(barwidth/2), df_dec_zero.median(), barwidth, label='_hidden', color=color_zero, linewidth=0)
                if (df_dec_zero.median() > 0):
                    axs[i].errorbar(x[j]+(barwidth/2), df_dec_zero.median(), yerr=df_dec_zero.std(), fmt='none', color='black', linestyle='none', capsize=2, elinewidth=1)
       

        xticks = np.insert(x, 0, 0, axis=0)

        axs[i].set_title(labels_wards[i], fontsize=12)
        axs[i].set_xticks(xticks, [0] + epsilons)
        axs[0].set_ylabel('Significantly Decreased', fontsize=10)
        axs[i].set_xlabel('Epsilon', fontsize=10)

        axs[i].yaxis.set_tick_params(labelbottom=True)
        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.margins(y=0, x=0)
    fig.subplots_adjust(top=0.86, wspace=0.1)
    fig.suptitle(f'Significantly Decreased Groups {title}- Median', fontsize=16, y=1.0)

    if (len(ward_names) <= 3):
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=10)
    else:
        plt.legend(bbox_to_anchor=(-1.8, -0.3), loc='lower center', ncol=3, borderaxespad=0, fontsize=10)

    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_median{fig_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{column_name}_experiments_geo_median{fig_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_data_error(df_populations, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False):
    plt.style.use('seaborn-v0_8-dark')
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    df_populations.sort_values(by=['total %'], inplace=True, ascending=False)


    for i, name in enumerate(ward_names):
        area_df = df_populations.query('area_name == @name & epsilon == 0.1')
    
        sns.boxplot(data=area_df, x='EthnicGroup', y='data error %', width=0.1, gap=0.2, color='black', showfliers=showfliers, fliersize=1, fill=False, linewidth=0.6, ax=axs[i])
        sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
        
        axs[i].tick_params(labelbottom=False) 
        axs[i].tick_params(axis='y', labelsize=8)

        axs[i].set_ylabel('Population Percentage')
        axs[i].set_xlabel('Population Group')
        axs[i].set_title(labels_wards[i], fontsize=12)

    fig.suptitle(f'Error Introduced by Data Error', fontsize=16, y=1.0)
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_data_error_boxen(df_populations, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False):
    plt.style.use('seaborn-v0_8-dark')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

    for i, name in enumerate(ward_names):
        
        area_df = df_populations.query('area_name == @name & epsilon == 0.1')
    
        sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
        sns.boxenplot(data=area_df, x='EthnicGroup', y='data error %', width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], flier_kws=dict(s=1.2))

        axs[i].tick_params(labelbottom=False) 
        axs[i].tick_params(axis='y', labelsize=8)

        axs[i].set_ylabel('Population Percentage')
        axs[i].set_xlabel('Population Group')
        axs[i].set_title(labels_wards[i], fontsize=12)

    fig.suptitle(f'Error Introduced by Data Error', fontsize=16, y=1.0)
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_boxen{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_boxen{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_data_error_numbers_boxen(df_populations, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False):
    plt.style.use('seaborn-v0_8-dark')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    df_populations.sort_values(by=['total %'], inplace=True, ascending=False)


    for i, name in enumerate(ward_names):

        area_df = df_populations.query('area_name == @name & epsilon == 0.1')
    
        sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
        sns.boxenplot(data=area_df, x='EthnicGroup', y='PopulationNumbersDataError', width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], flier_kws=dict(s=1.2))

        axs[i].tick_params(labelbottom=False) 
        axs[i].tick_params(axis='y', labelsize=8)

        axs[i].set_ylabel('Population Numbers')
        axs[i].set_xlabel('Population Group')
        axs[i].set_title(labels_wards[i], fontsize=12)

    fig.suptitle(f'Error Introduced by Data Error', fontsize=16, y=1.0)
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_numbers_boxen{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_numbers_boxen{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_data_error_numbers(df_populations, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False):
    plt.style.use('seaborn-v0_8-dark')

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    df_populations.sort_values(by=['total %'], inplace=True, ascending=False)


    for i, name in enumerate(ward_names):

        area_df = df_populations.query('area_name == @name & epsilon == 0.1')
    
        sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
        sns.boxplot(data=area_df, x='EthnicGroup', y='PopulationNumbersDataError', width=0.3, gap=0.2, color='black', showfliers=showfliers, fliersize=1.2, fill=False, linewidth=0.6, ax=axs[i])

        axs[i].tick_params(labelbottom=False) 
        axs[i].tick_params(axis='y', labelsize=8)

        axs[i].set_ylabel('Population Numbers')
        axs[i].set_xlabel('Population Group')
        axs[i].set_title(labels_wards[i], fontsize=12)

    fig.suptitle(f'Error Introduced by Data Error', fontsize=16, y=1.0)
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_numbers{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'data_error_population_numbers{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')

    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')


        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')
        
            sns.boxplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', showfliers=showfliers, fliersize=1.2, fill=False, linewidth=0.6, ax=axs[i])
            sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=lineval, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')


            axs[i].set_ylabel('Population Percentage')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Geometric Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_geo_{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_geo_{epsilon}{file_name}.svg')), bbox_inches='tight')

def population_plot_error_bars_boxen(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')

    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')


        for i, name in enumerate(ward_names):
            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')
        
            sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            sns.boxenplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], flier_kws=dict(s=1.2))
          
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=lineval, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')


            axs[i].set_ylabel('Population Percentage')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Geometric Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_geo_boxen{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_geo_boxen{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_laplace(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')
    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')
        
            sns.boxplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', showfliers=showfliers, fliersize=1.2, fill=False, linewidth=0.6, ax=axs[i])
            sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=lineval, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')

            axs[i].set_ylabel('Population Percentage')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Laplace Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_laplace_{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_laplace_{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_laplace_boxen(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')
    
    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')
        
            sns.barplot(data=area_df, x='EthnicGroup', y='total %', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            sns.boxenplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], flier_kws=dict(s=1.2))
        
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=lineval, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')

            axs[i].set_ylabel('Population Percentage')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Laplace Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_laplace_boxen{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_laplace_boxen{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_scatterplot(df_populations, column_name, xlabel, epsilons, colors, title, fig_folder_name, file_name):
    plt.style.use('seaborn-v0_8-dark')

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True, sharex=True)

    df_group = df_populations.groupby(['EthnicGroup', 'epsilon', 'area_name', 'total %'])[column_name].median().to_frame()
    df_group.reset_index(inplace=True)

    for i, epsilon in enumerate(epsilons):
        
        df_group.sort_values(by=['total %'], inplace=True, ascending=False)
        df = df_group.query('epsilon == @epsilon')
    
        p = sns.scatterplot(data=df, x=column_name, y='total %', hue='area_name', ax=axs[i], palette=colors, s=40, linewidth=0.3, edgecolor='black')
 
        axs[i].set_title(f'Epsilon = {epsilon}', fontsize=12)
        axs[0].set_ylabel('Population Percentage', fontsize=11,)

        p.legend().remove()
        p.set(xlabel=None)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, fontsize=11)
    fig.supxlabel(xlabel, fontsize=10)
    fig.subplots_adjust(top=0.85, wspace=0.1, bottom=0.12)

    fig.suptitle(title, fontsize=16, y=1.0)
    plt.savefig((os.path.join(fig_folder_name, f'{file_name}.png')), bbox_inches='tight')
    plt.savefig((os.path.join(fig_folder_name, f'{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_numbers(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')
    
    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')

            total = area_df.groupby(['EthnicGroup'])['PopulationNumbers'].median().sum()
            percentage = (lineval / 100 * total) 
        
        
            sns.boxplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', showfliers=showfliers, fliersize=1.2, fill=False, linewidth=0.6, ax=axs[i], 
                        medianprops={'linewidth': 1.4})
            sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=percentage, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')


            axs[i].set_ylabel('Population Numbers')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Geometric Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_geo_{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_geo_{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_numbers_boxen(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')
    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')

            total = area_df.groupby(['EthnicGroup'])['PopulationNumbers'].median().sum()
            percentage = (lineval / 100 * total) 
        
            sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            sns.boxenplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], 
                          flier_kws=dict(s=1.2), line_kws=dict(linewidth=1.3))
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=percentage, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')

            axs[i].set_ylabel('Population Numbers')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Geometric Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_geo_boxen_{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_geo_boxen_{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_numbers_laplace(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')
    
    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')

            total = area_df.groupby(['EthnicGroup'])['PopulationNumbers'].median().sum()
            percentage = (lineval / 100 * total) 

            sns.boxplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', showfliers=showfliers, fliersize=1.2, fill=False, linewidth=0.6, ax=axs[i], medianprops={'linewidth': 1.4})
            sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=percentage, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')

            axs[i].set_ylabel('Population Numbers')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Laplace Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_laplace_{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_laplace_{epsilon}{file_name}.svg')), bbox_inches='tight')


def population_plot_error_bars_numbers_laplace_boxen(df_populations, column_name, epsilons, ward_names, labels_wards, colors, fig_folder_name, title="", file_name="", showfliers=False, symlog=True, lineval=0):
    plt.style.use('seaborn-v0_8-dark')

    for epsilon in epsilons:

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        df_populations.sort_values(by=['total %'], inplace=True, ascending=False)

        if (symlog):
            plt.yscale('symlog')

        for i, name in enumerate(ward_names):

            area_df = df_populations.query('area_name == @name & epsilon == @epsilon')

            total = area_df.groupby(['EthnicGroup'])['PopulationNumbers'].median().sum()
            percentage = (lineval / 100 * total) 
        
            sns.barplot(data=area_df, x='EthnicGroup', y='PopulationNumbers', width=1.1, gap=0.2, color=colors[0], linewidth=0, ax=axs[i])
            sns.boxenplot(data=area_df, x='EthnicGroup', y=column_name, width=0.3, gap=0.2, color='black', fill=False, linewidth=0.6, ax=axs[i], 
                          flier_kws=dict(s=1.2), line_kws=dict(linewidth=1.3))
            
            axs[i].tick_params(labelbottom=False) 
            axs[i].tick_params(axis='y', labelsize=8)

            if (lineval > 0): 
                line = axs[i].axhline(y=percentage, linewidth=0.9, color='black', linestyle='--')
                y = line.get_ydata()[-1]
                axs[i].annotate(f'{lineval}%', xy=(1,y), xytext=(5,0), color=line.get_color(), 
                xycoords = axs[i].get_yaxis_transform(), textcoords='offset points',
                size=10, va='center')

            axs[i].set_ylabel('Population Numbers')
            axs[i].set_xlabel('Population Group')
            axs[i].set_title(labels_wards[i], fontsize=12)

        fig.suptitle(f'Error Introduced with Laplace Mechanism with {title}Epsilon = {epsilon}', fontsize=16, y=1.0)
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_laplace_boxen{epsilon}{file_name}.png')), bbox_inches='tight')
        plt.savefig((os.path.join(fig_folder_name, f'{column_name}_population_numbers_laplace_boxen{epsilon}{file_name}.svg')), bbox_inches='tight')


def make_experiment_metric_df(experiments):
    df_list = []

    for i in range(len(experiments)):
        metrics_df = experiments[i]['metrics_df'].copy(deep=True)
        df_list.append(metrics_df)
        metrics_df.reset_index(inplace=True)

    dfs = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    dfs['lowest_observed'] = dfs.groupby(['epsilon', 'area_name'])['total_dp'].transform('min')
    dfs['largest_observed'] = dfs.groupby(['epsilon', 'area_name'])['total_dp'].transform('max')

    return dfs


def create_experiments_table_pop(experiments, table_features): 
    metrics_df = make_experiment_metric_df(experiments)
    df_median = metrics_df.groupby(['epsilon', 'area_name'])[table_features].median()
    latex_table = df_median.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)
    

def create_experiments_table_pop_merged(experiments, experiments_clip, table_features, table_features_clip): 
    metrics_df = make_experiment_metric_df(experiments)
    metrics_clip_df = make_experiment_metric_df(experiments_clip)
    df_median = metrics_df.groupby(['epsilon', 'area_name'])[table_features].median()
    df_median_clip = metrics_clip_df.groupby(['epsilon', 'area_name'])[table_features_clip].median()
    df_median_clip.columns = [s + '_clip' for s in table_features_clip]

    table_df = pd.concat([df_median, df_median_clip], axis=1)
    latex_table = table_df.to_latex(index=True, multirow=True, escape=False, float_format='{:0.0f}'.format)
    print(latex_table)
