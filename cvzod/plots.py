import csv
from pathlib import Path

import numpy as np

from .io import write_json
from .config import METHODS, figure_for, run_key

LABELS = dict(zip(METHODS, ('CV-ZOD', 'ZOD', 'GES', 'SurGD')))
COLORS = dict(zip(METHODS, ('#a78bfa', '#10b981', '#ef4444', '#f97316')))
HISTORY_COLORS = {0:'#303030', 1:'#9556a5', 2:'#2879b7', 4:'#25866b', 8:'#d47720'}
HISTORY_STYLES = {0:'-', 1:'--', 2:'-.', 4:':', 8:(0,(5,1,1,1))}


def moments(values):
    values = np.asarray(values, dtype=float)
    assert values.shape[0] >= 1 and np.isfinite(values).all()
    sd = values.std(axis=0, ddof=1).tolist() if len(values) > 1 else None
    return dict(mean=values.mean(axis=0).tolist(), sample_sd=sd, sample_count=len(values))


def objective(run):
    return np.r_[run['iterations'][0]['objective_before'], [row['objective_after'] for row in run['iterations']]]


def cosines(run):
    return np.asarray([row['cosine'] for row in run['diagnostics']])


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(figure, output, name):
    figure.savefig(output/(name+'.pdf'), dpi=300)
    import matplotlib.pyplot as plt
    plt.close(figure)


def comparison_series(values):
    series = []
    for index, k in enumerate(values):
        series.append(dict(key='cv-zod' if k == 1 else f'cv-zod-k{k}', method='cv-zod', k=k,
                           label='CV-ZOD' if values == [1] else f'CV-ZOD (k={k})',
                           color=COLORS['cv-zod'] if k == 1 else HISTORY_COLORS.get(k, f'C{(index+4)%10}'),
                           style='-' if k == 1 else HISTORY_STYLES.get(k, ('--', '-.', ':')[index%3])))
    series.extend(dict(key=method, method=method, k=None, label=LABELS[method],
                       color=COLORS[method], style=':' if method == 'surgd' else '-')
                  for method in ('zod', 'ges', 'surgd'))
    return series


def draw_band(axis, statistics, color, label=None, linestyle='-', alpha=.15, linewidth=1.7):
    mean = np.asarray(statistics['mean'])
    x = np.arange(len(mean))
    if statistics['sample_sd'] is not None:
        sd = np.asarray(statistics['sample_sd'])
        axis.fill_between(x, mean-sd, mean+sd, color=color, alpha=alpha, linewidth=0)
    axis.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=linewidth,
              marker='o' if len(mean) == 1 else None)


def figure2(config, runs, output):
    import matplotlib.pyplot as plt
    seeds = config['fluid_seeds']
    steps = config['iterations']
    values = config['hint_dims']['fluid']
    series = comparison_series(values)
    curves = {}
    for entry in series:
        method, key = entry['method'], entry['key']
        selected = [runs[run_key('fluid', method, seed, entry['k'])] for seed in seeds]
        curves[key] = dict(objective=moments([objective(run) for run in selected]))
        if method in ('cv-zod','ges'):
            curves[key]['cosine'] = moments([cosines(run) for run in selected])
    rows = (len(series)+3)//2
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42}):
        figure = plt.figure(figsize=(6.7,max(3.35,1.1*rows)))
        upper = figure.add_axes([.085,1.1465/3.35,.5775,2.0085/3.35])
        lower = figure.add_axes([.085,.39575/3.35,.5775,.47775/3.35],sharex=upper)
        for entry in series:
            draw_band(upper,curves[entry['key']]['objective'],entry['color'],entry['label'],entry['style'])
            if entry['method'] in ('cv-zod','ges'):
                draw_band(lower,curves[entry['key']]['cosine'],entry['color'],linestyle=entry['style'])
        upper.set(ylabel='Objective',ylim=(0,None));upper.tick_params(labelbottom=False)
        upper.legend(loc='upper right',framealpha=.95,ncol=2,columnspacing=1.2,handlelength=2.6,fontsize=8.5)
        lower.set(xlabel='Iteration',ylabel=r'$\cos(\theta_t)$',ylim=(-.025,1.025),xlim=(0,steps))
        lower.set_yticks((0,.5,1));lower.set_xticks(np.linspace(0,steps,6))
        for axis in (upper,lower):
            axis.grid(alpha=.2);axis.set_axisbelow(True)
        first = seeds[0]
        with np.load(output/'data'/(run_key('fluid', 'cv-zod', first, values[0])+'.npz')) as archive:
            panels = [(archive['initial_density'],'Initial density'),(archive['target_density'],'Target density')]
        for entry in reversed(series):
            method = entry['method']
            key = run_key('fluid', method, first, entry['k'])
            value = runs[key]['final_objective']
            title = f'CV-ZOD\nk={entry["k"]} ({value:.4f})' if method == 'cv-zod' and values != [1] else f'{entry["label"]} ({value:.4f})'
            with np.load(output/'data'/(key+'.npz')) as archive:
                panels.append((archive['terminal_density'],title))
        grid = figure.add_gridspec(rows,2,left=.7325,right=.995,bottom=.08,top=.94,wspace=.10,
                                  hspace=.30 if values == [1] else .45)
        for index,(density,title) in enumerate(panels):
            axis = figure.add_subplot(grid[index//2,index%2])
            axis.imshow(density,cmap='viridis',origin='lower',vmin=0,vmax=1,interpolation='nearest')
            axis.set_title(title,fontsize=7.1,color='black',pad=3.5)
            axis.set_xticks([]);axis.set_yticks([])
            for spine in axis.spines.values():spine.set_color('#333333');spine.set_linewidth(.65)
        save(figure,output,'figure2')
    write_json(output/'figure2.json',dict(sample_count=len(seeds),seeds=seeds,k=values,
               reconstruction_seed=first,reconstruction_selection='first prespecified evaluation seed',
               uncertainty='mean +/- one sample standard deviation; ddof=1' if len(seeds) > 1 else 'single run; no uncertainty band',curves=curves))


def figure3(config, runs, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    molecules = ('efavirenz','adenosine','benzylpenicillin')
    steps = config['iterations']
    count = config['conformations']
    hint_dims = config['hint_dims']['molecules']
    series = comparison_series(hint_dims)
    curves,atoms = {},{}
    all_values = []
    for molecule in molecules:
        curves[molecule] = {}
        for entry in series:
            method, key = entry['method'], entry['key']
            selected = sorted([run for run in runs.values() if run.get('molecule')==molecule and run['settings']['method']==method
                               and (method != 'cv-zod' or run['settings']['history']==entry['k'])],key=lambda r:r['seed'])
            assert len(selected)==count
            energies = [objective(run)-objective(run)[0] for run in selected]
            statistics = dict(energy_change=moments(energies))
            all_values.append(statistics['energy_change']['mean'])
            if method in ('cv-zod','ges'):statistics['cosine']=moments([cosines(run) for run in selected])
            curves[molecule][key] = statistics
            atoms[molecule]=len(selected[0]['provenance']['symbols'])
    energy_limits=(float(np.floor(np.min(all_values)*2)/2),float(np.ceil(np.max(all_values)*2)/2+.05))
    legend_extra=.24*((len(series)+3)//4-1)
    combined_height=3.5-(3.5*(.87-.17)/(1+.18/2))*.5+legend_extra
    style={'font.family':'DejaVu Sans','font.size':7.7,'axes.titlesize':8,'axes.labelsize':7.7,
           'xtick.labelsize':7.2,'ytick.labelsize':7.2,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6}
    with plt.rc_context(style):
        figure,axes=plt.subplots(2,3,figsize=(6.3,combined_height),sharex='col',sharey='row',
                                 gridspec_kw={'height_ratios':(2.0085,.47775)})
        for column,molecule in enumerate(molecules):
            for entry in series:
                values=curves[molecule][entry['key']]['energy_change']['mean']
                axes[0,column].plot(np.arange(len(values)),values,color=entry['color'],linestyle=entry['style'],lw=1.5)
                if entry['method'] in ('cv-zod','ges'):
                    values=curves[molecule][entry['key']]['cosine']['mean']
                    axes[1,column].plot(np.arange(len(values)),values,color=entry['color'],linestyle=entry['style'],lw=1.5,
                                        marker='o' if len(values) == 1 else None)
            axes[0,column].set_title(f'{molecule.capitalize()}\n$N={atoms[molecule]}$')
            axes[0,column].set_ylim(*energy_limits);axes[0,column].tick_params(labelbottom=False)
            axes[1,column].set(xlabel='Iteration',ylim=(0,1));axes[1,column].set_yticks([0,.5,1])
            for axis in axes[:,column]:
                axis.axhline(0,color='#777777',lw=.55,zorder=0);axis.grid(alpha=.18,lw=.5)
                axis.set_xlim(0,steps);axis.set_xticks([0,steps/2,steps]);axis.spines[['top','right']].set_visible(False)
                axis.tick_params(length=2.5,width=.5)
        axes[0,0].set_ylabel('Energy change (eV)');axes[1,0].set_ylabel(r'$\cos(\theta_t)$')
        handles=[Line2D([],[],color=entry['color'],linestyle=entry['style'],lw=1.5,label=entry['label']) for entry in series]
        figure.legend(handles=handles,loc='lower center',ncol=min(4,len(series)),frameon=False,bbox_to_anchor=(.53,.005),columnspacing=2)
        figure.subplots_adjust(left=.082,right=.975,top=1-3.5*(1-.87)/combined_height,
                               bottom=(3.5*.17+legend_extra)/combined_height,wspace=.17,hspace=.36)
        save(figure,output,'figure3')
    write_json(output/'figure3.json',dict(conformations_per_molecule=count,k=hint_dims,curves=curves))


def figure4(config, runs, output):
    import matplotlib.pyplot as plt
    seeds = config['fluid_seeds']
    steps = config['iterations']
    lengths = config['hint_dims']['fluid']
    maximum = max(lengths)
    metric = f'common_k{maximum}_cosine'
    keys_for = lambda k: [run_key('fluid', 'zod' if k == 0 else 'cv-zod', seed, k) for seed in seeds]
    common = [runs[key]['diagnostics'] for key in keys_for(maximum)]
    curves = {}
    for k in (0, *lengths):
        keys = keys_for(k)
        selected = [runs[key] for key in keys]
        curves[str(k)] = dict(objective=moments([objective(run) for run in selected]), source_runs=keys)
        if k:
            curves[str(k)]['own_trajectory_cosine'] = moments([cosines(run) for run in selected])
    for k in lengths:
        curves[str(k)][metric] = moments([[row['cosine'] if k == maximum else row['nested_cosines'][str(k)] for row in entries]
                                         for entries in common])
    style = {'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
             'legend.fontsize':7.5, 'pdf.fonttype':42}
    with plt.rc_context(style):
        colors = {k:HISTORY_COLORS.get(k, plt.get_cmap('tab10')(index % 10))
                  for index, k in enumerate((0, *lengths))}
        styles = {k:HISTORY_STYLES.get(k, ('-', '--', '-.', ':')[index % 4])
                  for index, k in enumerate((0, *lengths))}
        figure, axes = plt.subplots(2, 1, figsize=(5.5, 2.65), sharex=True,
                                   gridspec_kw={'height_ratios':(2.0085, .47775)}, layout='constrained')
        for k in (0, *lengths):
            draw_band(axes[0], curves[str(k)]['objective'], colors[k],
                      'ZOD' if k == 0 else f'$k={k}$', styles[k], alpha=.09, linewidth=1.5)
        for k in lengths:
            draw_band(axes[1], curves[str(k)][metric], colors[k],
                      linestyle=styles[k], alpha=.09, linewidth=1.5)
        axes[0].set(ylabel='Objective', xlim=(0, steps), ylim=(0, None))
        axes[0].tick_params(labelbottom=False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside upper center', ncol=min(5, len(curves)), frameon=False,
                      columnspacing=.8, handlelength=1.7, handletextpad=.4, fontsize=7)
        axes[1].set(xlabel='Iteration', ylabel=r'$\cos(\theta_t)$', xlim=(0, steps), ylim=(0, 1))
        axes[1].set_yticks([0, .5, 1])
        axes[1].set_xticks(np.linspace(0, steps, 6))
        figure.align_ylabels(axes)
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(alpha=.15, lw=.5)
        save(figure, output, 'figure4')
    rows = []
    for seed in seeds:
        for k in lengths:
            own = runs[run_key('fluid', 'cv-zod', seed, k)]['diagnostics']
            reference_run = runs[run_key('fluid', 'cv-zod', seed, maximum)]['diagnostics']
            for step, (current, reference) in enumerate(zip(own, reference_run)):
                cosine = reference['cosine'] if k == maximum else reference['nested_cosines'][str(k)]
                rows.append(dict(seed=seed, pre_update_iteration=step, history_length=k,
                                 own_cosine=current['cosine'], own_angle_degrees=current['angle_degrees'],
                                 **{metric:cosine,
                                    f'common_k{maximum}_angle_degrees':float(np.degrees(np.arccos(cosine)))}))
    write_csv(output/'figure4_angles.csv', rows)
    write_json(output/'figure4.json', dict(sample_count=len(seeds), seeds=seeds,
               diagnostic_iterations=list(range(steps)), k=list(lengths),
               common_history_length=maximum, curves=curves))


def render(experiment, config, runs, output):
    figure = figure_for(experiment, config)
    {2:figure2, 3:figure3, 4:figure4}[figure](config, runs, Path(output))
