### xCT metabolomics analysis
## Oct. 31st, 2024

# import modules
import pandas as pd
import numpy as np
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from pyopls import OPLS


# define methods for plotting

# define a simple method for labeling datapoints in our scatter plot based on their dataframe index
def label_scatter(ax, x, y, data, to_label=None):
	"""
	ax = matplotlib axes object of our scatter plot
	x = x-axis column name
	y = y-axis column name
	data = pandas dataframe
	to_label = optional list of names we want to label
	"""

	# Label the desired points
	if to_label:
		for name in to_label:
			ax.text(data.loc[name, x]+0.02, data.loc[name, y]+0.02, name)

	else:
		# If not, for each row in our data, add a label to the plot axis near the
		# corresponding data point.
		n = 0
		for name, vals in data.iterrows():
			if n % 2 == 0:
				ax.text(vals[x]+0.02, vals[y]+0.02, name)
				n += 1
			else:
				n += 1
	return

# define a method for identifying significant metabolites and differential abundance as volcano plot
def volcano(data, x, y, x_thresh=1, y_thresh=0.05, sig_palette=None):
	"""
	data = input metabolite abundance dataframe (metabolites as rows)
	x = Column name containg fold change values
	y = Column name containing p-values
	x_thresh = significance threshold for fold change values
	y_thresh = significance threshold for p-values
	sig_palette = list of colour values for [negative metab, postive metab]
	"""

	# Want to identify the significant metabs according to our x_thresh, y_thresh vals
	pos_metabs = data.where((data[y] < y_thresh) & (data[x] >= abs(x_thresh)))
	pos_metabs.dropna(axis=0, how='all', inplace=True)
	neg_metabs = data.where((data[y] < y_thresh) & (data[x] <= -1*abs(x_thresh)))
	neg_metabs.dropna(axis=0, how='all', inplace=True)
	print(f'pos_metabs:\n{pos_metabs}\nneg_metabs:\n{neg_metabs}')

	# define gene groups columns for colouring volcano plot points
	data['Differential abundance'] = 'Non-significant'
	data.loc[pos_metabs.index, 'Differential abundance'] = 'Positive Differential metabolite'
	data.loc[neg_metabs.index, 'Differential abundance'] = 'Negative Differential metabolite'
	print(f'pos metabs:\n{data.loc[pos_metabs.index, "Differential abundance"]}\n\nne metabs:\n{data.loc[neg_metabs.index, "Differential abundance"]}')
	data['-log10 p-value'] = np.log10(data[y])*-1   # format values for y-axis

	# Define colour palette for plot
	if sig_palette:
		plot_palette = {
			'Negative Differential metabolite': sig_palette[0], 
			'Non-significant': 'darkgrey',
			'Positive Differential metabolite': sig_palette[1]
			}
	else:
		plot_palette = {
			'Negative Differential metabolite': 'royalblue', 
			'Non-significant': 'darkgrey',
			'Positive Differential metabolite': 'tomato'
			}

	# # palette is coloured according to their appearance in the dataset
	# # Can guarentee that our ordering is correct by sorting the data values before plotting
	# sort_data = data.sort_values(by=x, ascending=True)

	# plot the data
	volc = sns.scatterplot(data=data, 
			x=x, 
			y='-log10 p-value',
			hue='Differential abundance',
			alpha=0.90,
			linewidth=0,
			s=40,
			palette=plot_palette
			)

	# Set plot lims
	volc.set_xlim([-1.5,1.5])
	volc.set_ylim([-0.05,data['-log10 p-value'].max()+0.5])

	# Add lines corresponding to significance thresholds
	ymin, ymax = volc.get_ylim()
	xmin, xmax = volc.get_xlim()

	volc.vlines(x=[-1*abs(x_thresh),abs(x_thresh)], ymin=ymin, ymax=ymax, colors='grey', ls='--', lw=0.75)
	volc.hlines(y=-1*np.log10(y_thresh), xmin=xmin, xmax=xmax, colors='grey', ls='--', lw=0.75)

	plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')

	return volc

# define method for computing VIP scores for PLS-DA
# from https://github.com/scikit-learn/scikit-learn/issues/7050
def VIP(model):
	t = model.x_scores_
	w = model.x_weights_ # replace with x_rotations_ if needed
	q = model.y_loadings_ 
	features_, _ = w.shape
	vip = np.zeros(shape=(features_,))
	inner_sum = np.diag(t.T @ t @ q.T @ q)
	SS_total = np.sum(inner_sum)
	vip = np.sqrt(features_*(w**2 @ inner_sum)/ SS_total)
	return vip



# load data
metab_df = pd.read_csv(r'unlabeled_metabolomics_prl_missing.csv',index_col=0) 	## NOTE: Replace with the correct name/path for the unlabeled metabolomics file on your system.
print(f'metab_df:\n{metab_df}')
metab_df = metab_df.iloc[:-3,:]     # remove metadata at end

# get by group
metab_1 = metab_df.index[1]     # want to skip the 'group' row in our subset df
print(f'metab_1:\n{metab_1}')
sut_df = metab_df.loc[metab_1:,metab_df.loc['group'] == 'sut']
wt_df = metab_df.loc[metab_1:,metab_df.loc['group'] == 'wild-type']
print(f'sut_df:\n{sut_df.dtypes}, {wt_df.dtypes}')
sut_df = sut_df.astype('float')
wt_df = wt_df.astype('float')

# Compute Welch's t-test between metabolite levels in sut and WT groups
ttest = scipy.stats.ttest_ind(sut_df, wt_df, axis=1, equal_var=False, nan_policy='omit')
print(f'ttest:\n{ttest}')
for i in ttest:
	print(f'{i}\n {i.shape}')

# Include p-values as column
temp = np.insert(ttest[1], 0, np.nan)   # insert empty value for initial 'group' var
print(f'temp:\n{temp}\n{ttest[1]}')
metab_df['t-test p-value'] = temp
print(metab_df)
sig_metabDF = metab_df.loc[metab_df['t-test p-value'] < 0.05]
print(sig_metabDF)


# Compute median log2 fold change between groups
sut_med = sut_df.median(axis=1)
wt_med = wt_df.median(axis=1)

metab_FC =  wt_med / sut_med
print(f'metab_FC:\n{metab_FC}')
metab_df['log2 FC'] = np.log2(metab_FC)
# metab_df.to_csv(rf'xCT_unlabeled_diffMetab.csv')
# exit()
# print(f'metab_df:\n{metab_df}')

# Define colours for plotting
colours = [(7/255,126/255,151/255), (225/255,128/255,0/255)] 	# RGB values /255 for seaborn plotting
print(f'colours:\n{colours}')

differential_metab = volcano(data=metab_df, 
      x='log2 FC', 
      y='t-test p-value', 
      x_thresh=0.5,   # specific our log2 FC threshold (absolute)
      y_thresh=0.05,  # specify our p-value threshold
      sig_palette=colours  # colours for plotting
      )


# print metab_df to see if its modified
print(f'metab_df after plotting:\n{metab_df}')

# Format the plot such that we label our significant data points
sig_data = metab_df.loc[metab_df['Differential abundance'] != 'Non-significant']
print(f'sig_data:\n{sig_data}')
print(f'sig_data[diff abund]:\n{sig_data["Differential abundance"]}')

# Define some points to label
label_metabs = ['Proline', 'Glycine', 'Methionine', 'Leucine', 'Citrulline', 'Glutathione', 'Uracil', 'Orotic acid', 'Glutamine']
label_scatter(differential_metab, 'log2 FC', '-log10 p-value', sig_data, to_label=label_metabs)

# rename axes
plt.xlabel('Median log$_2$ fold change')
plt.ylabel('-log$_{10}$ p-value')
plt.title(r'Wild-type / Slc7a11$^{\mathit{sut}/\mathit{sut}}$ (median abundance)')

fig = differential_metab.get_figure()
fig.savefig(f'xCT_metabolomics_volcano_labeled.tiff', bbox_inches='tight', dpi=600)
plt.close()
exit()

# For each set of samples, we want to cluster the metabolites to observe any distances in clustering

# clustering method cannot handle missing values, replace these with a limit of detection value
# Define limit of detection as 1/5 minimum feature value across ALL samples
all_metabDF = pd.concat([sut_df, wt_df],axis=1)
lods = all_metabDF.min(axis=1) * 0.2

impAll_metabDF = all_metabDF.T.replace(np.nan, lods).T

# Due to the differences in relative metabolite abundances, normalize values for clustering
scaler = StandardScaler()   # Z-score normalization
z_AllMetabDF = scaler.fit_transform(impAll_metabDF.T).T

# split data into groups
print(f'z_AllMetabDF:\n{z_AllMetabDF}, shape:{z_AllMetabDF.shape}')

z_sutDF = z_AllMetabDF[:,:6]
z_wtDF = z_AllMetabDF[:,6:]

print(f'Initial values vs imputed values for sut')
print(f'z_sutDF:\n{z_sutDF}, shape:{z_sutDF.shape}')
print(f'z_wtDF:\n{z_wtDF}, shape:{z_wtDF.shape}')


sut_dist = pdist(z_sutDF, 'euclidean')
sut_links = hierarchy.ward(sut_dist)

wt_dist = pdist(z_wtDF, 'euclidean')
wt_links = hierarchy.ward(wt_dist)

# hierarchy.dendrogram(sut_links, labels=all_metabDF.index)
# plt.savefig('temp_sut_clustering_06-11-24.png', dpi=300, bbox_inches='tight')
# plt.close()

# hierarchy.dendrogram(wt_links, labels=all_metabDF.index)
# plt.savefig('temp_wt_clustering_06-11-24.png', dpi=300, bbox_inches='tight')
# plt.close()

# also compute with log10 values
log_impAll_metabDF = np.log10(impAll_metabDF)

logZ_All_metabDF = scaler.fit_transform(log_impAll_metabDF.T).T

logZ_sutDF = logZ_All_metabDF[:,:6]
logZ_wtDF = logZ_All_metabDF[:,6:]


logSut_links = hierarchy.linkage(logZ_sutDF, 'ward', optimal_ordering=True)

logWT_links = hierarchy.linkage(logZ_wtDF, 'ward', optimal_ordering=True)
print(f'logWT_links:\n{logWT_links}')

# Define plot objects
fig, axes = plt.subplots(1, 2, figsize=(14, 16))
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

axes[0].set_title('Wild-type', fontsize=16)
dend_wt = hierarchy.dendrogram(logWT_links, ax=axes[0], labels=all_metabDF.index, orientation='right', count_sort=True, leaf_font_size=10, color_threshold=4)

# Colour labels the same as their corresponding cluster
# Get the RGB/hex values for 'C1', 'C2', etc.
colour_map = {f'C{i}': default_colours[i] for i in range(len(default_colours))}

# colour labels
label_colours = {}
for colour, leaf_label in zip(dend_wt['leaves_color_list'], dend_wt['ivl']):
	label_colours[leaf_label] = colour_map[colour]

y_labels = axes[0].get_ymajorticklabels()  # Get the labels on the x-axis

# Assign label colours
for label in y_labels:
	label_text = label.get_text()
	if label_text in label_colours:
		label.set_color(label_colours[label_text])  # Set the color based on the cluster

# Repeat for Sut samples
dend_sut = hierarchy.dendrogram(logSut_links, ax=axes[1], labels=all_metabDF.index, orientation='left', count_sort=True, leaf_font_size=10, color_threshold=4)
axes[1].set_title(r'Slc7a11$^{sut/sut}$', fontsize=16)

label_colours = {}
for colour, leaf_label in zip(dend_sut['leaves_color_list'], dend_sut['ivl']):
	label_colours[leaf_label] = colour_map[colour]

y_labels = axes[1].get_ymajorticklabels()  # Get the labels on the x-axis

# Assign label colours
for label in y_labels:
	label_text = label.get_text()
	if label_text in label_colours:
		label.set_color(label_colours[label_text])  # Set the color based on the cluster


plt.savefig('xct_clusteringD4_27-11-24.tiff', dpi=600, bbox_inches='tight')
# exit()

# For a secondary figure, we would like to display clusters with differential metabolites
# between Slc7a11 mutant and WT  

DIFFERENTIAL_METABOLITES = [
	'Leucine','Tryptophan',
	'Phenylalanine','Proline',
	'Citrulline','Isoleucine',
	'Ethyl pyruvate','Uridine',
	'Tyrosine','Glycine',
	'Valine','Methionine',
	'Histidine','Threonine',
	'Histamine','Ornithine',
	'Glutamine','Asparagine',
	'Serine','Malonic acid',
	'2-Acetamido-2-Deoxy-D-Glucopyranose','Orotic acid',
	'Glutathione','Uracil',
	'4,5-dihydroorotate','Dihydrouracil',
	'Lysine'
]

# As a minor formatting improvement, we can add * to each metabolite of interest to further clarify
# Adjust these labels in each dendrogram
for i, ax in enumerate(axes):
	ax_labels = [label.get_text() for label in ax.get_ymajorticklabels()]
	new_labels = []
	for text in ax_labels:
		if text in DIFFERENTIAL_METABOLITES:
			if i % 2 == 0:
				new_text = rf'★    {text}'
			else:
				new_text = rf'{text}    ★'

		else:
			new_text = text
		new_labels.append(new_text)

	# Update the labels
	ax.set_yticklabels(new_labels)

	# Now need to bold the desired labels
	for label in ax.get_ymajorticklabels():
		if label.get_text().strip('★ ').strip() in DIFFERENTIAL_METABOLITES:
			label.set_fontweight('bold')  # Set bold font

fig.canvas.draw_idle() 	# Need to re-draw to update the label text
plt.savefig('xct_diff_clusteringD4_27-11-24.tiff', dpi=600, bbox_inches='tight')
exit()

# A final plot to colour each metabolite by the OTHER cluster to visualize shifts in clustering
# colour labels

dends = [dend_wt, dend_sut]     # Because the order is opposite to the plotted axis,
								# we end up colouring each dendrogram by the OTHER dend's clusters
for i, dend in enumerate(dends):
	label_colours = {}
	for colour, leaf_label in zip(dend['leaves_color_list'], dend['ivl']):
		label_colours[leaf_label] = colour_map[colour]

	y_labels = axes[i].get_ymajorticklabels()  # Get the labels on the x-axis

	# Assign label colours
	for label in y_labels:
		label_text = label.get_text()
		if label_text in label_colours:
			label.set_color(label_colours[label_text])  # Set the color based on the cluster

# plt.savefig('OTHERcolours_clusteringD4_15-11-24.png', dpi=600, bbox_inches='tight')
# exit()

metab_clusters = [pd.Series(data=x['leaves_color_list'], index=x['ivl']) for x in [dend_sut, dend_wt]]

METABOLITES = ['Proline', 'Serine', 'Glutamine']
sut_leaves = np.array(dend_sut['ivl'])
subset_positions = np.where(np.isin(sut_leaves, METABOLITES))[0]
print(rf'subset_positions: {subset_positions}')
sut_leave_coords = np.array(dend_sut['icoord'])
print(f'icoords gives positions: {sut_leave_coords}, shape: {sut_leave_coords.shape}')
sub_coords = sut_leave_coords[subset_positions]
print(f'sub_coords gives positions: {sub_coords}, shape: {sub_coords.shape}')
print(f'nth dim: {sub_coords[:,0]}, shape: {sub_coords[:,0].shape}')
print(f'nth dim min: {min(sub_coords[:,0])}, max: {max(sub_coords[:,0])}')

# plt.close()
# fig, ax = plt.subplots(1, 1)
# dend_sut = hierarchy.dendrogram(logSut_links, ax=ax, labels=all_metabDF.index, orientation='right', count_sort=True, leaf_font_size=12, color_threshold=4)
# ax.set_ylim(min(sub_coords[:,0])-50, max(sub_coords[:,0])-50)
# plt.savefig('temp_sut_log_clusteringD4_06-11-24_subset.png', dpi=300, bbox_inches='tight')
# plt.close()

# exit()

cluster_df = pd.concat(metab_clusters,axis=1)
print(f'cluster_df:\n{cluster_df}')
cluster_df.columns = ['sut clusters', 'wt clusters']
print(f'cluster_df:\n{cluster_df}')

# cluster_df.to_csv(rf'xct_metab_clusters_d4_06-11-24.csv')

# Generate a tanglegram of our clustering to illustrate different clusters for some metabolites of interest
metab_clusts = cluster_df.loc[METABOLITES]
print(metab_clusts)
sut_clusts = metab_clusts.loc[:,'sut clusters'].unique()
wt_clusts = metab_clusts.loc[:,'wt clusters'].unique()

print(sut_clusts)
print(wt_clusts)


wt_clustMetabs = cluster_df.loc[cluster_df['wt clusters'].isin(wt_clusts)].index
sut_clustMetabs = cluster_df.loc[cluster_df['sut clusters'].isin(sut_clusts)].index
print(wt_clustMetabs)
print(sut_clustMetabs)

# To further compare our clustering, we can quantify differences between the dendrograms
# Compute Robinson-Fould metric via ETE toolkit

# First format our dendrogram object into one usable by ete
sut_tree = hierarchy.to_tree(logSut_links, rd=False)
wt_tree = hierarchy.to_tree(logWT_links, rd=False)

# Convert tree data to newick text format
def to_newick(tree_node, leaf_labels, sep=""):

	# If we are at a leaf, return the label
	if tree_node.is_leaf():
		return f'\"{leaf_labels[tree_node.id]}\"{sep}'
	
	# Else, repeat until we are at a leaf node 
	else:
		left_newick = to_newick(tree_node.left, leaf_labels)
		right_newick = to_newick(tree_node.right, leaf_labels)
		return f'({left_newick},{right_newick}){sep}'


# Convert to Newick
labels = all_metabDF.index.tolist()
sut_newick = to_newick(sut_tree, labels) + ';'
wt_newick = to_newick(wt_tree,labels) + ';'


# Now we can use these trees in ETE to compute our metrics
from ete3 import Tree
sut_ete = Tree(sut_newick, format=8, quoted_node_names=True)
wt_ete = Tree(wt_newick, format=8, quoted_node_names=True)
wt_ete.render('test_wt_tree.png')
# Compute Robinson-foulds distance between trees
tree_comp = sut_ete.compare(wt_ete)
for j, result in enumerate(tree_comp):
	key = list(tree_comp.keys())
	print(f'Result - {key[j]}')
	print(tree_comp[key[j]])

# interested in normalized distance:
print(f'Normalized RF distance: {tree_comp["norm_rf"]}')

# Visually, the dendrogram for mutant samples appear to be clustered more tightly than wild-type
# We can compute the average linkage distances in each to quantify this

mean_sutDist = logSut_links[:,2].mean()
med_sutDist = np.median(logSut_links[:,2])
mean_wtDist = logWT_links[:,2].mean()
med_wtDist = np.median(logWT_links[:,2])

sd_sutDist = np.std(logSut_links[:,2])
sd_wtDist = np.std(logWT_links[:,2])
sem_sutDist = scipy.stats.sem(logSut_links[:,2])
sem_wtDist = scipy.stats.sem(logWT_links[:,2])

print(f"Mean & median linkage distance - Sut: {mean_sutDist}, {med_sutDist}")
print(f'\t+/- S.D. of {sd_sutDist}, SEM of {sem_sutDist}')
print(f"Mean & median linkage distance - WT: {mean_wtDist}, {med_wtDist}")
print(f'\t+/- S.D. of {sd_wtDist}, SEM of {sem_wtDist}')

exit()





# Tanglegram produced from subset is not clustering the same
# Need to create clustering from all metabolites, and display the susbset of clusters
logSut_links = hierarchy.linkage(logZ_sutDF, 'ward', optimal_ordering=True)

logWT_links = hierarchy.linkage(logZ_wtDF, 'ward', optimal_ordering=True)
print(f'logWT_links:\n{logWT_links}, shape:{logWT_links.shape}')
exit()

fig, axes = plt.subplots(1, 2, figsize=(14, 16))

wt_palette = sns.color_palette("Blues", n_colors=8, as_cmap=False)
sut_palette = sns.color_palette("Blues", n_colors=8, as_cmap=False)

dend_sut = hierarchy.dendrogram(logSut_links, ax=axes[0], labels=all_metabDF.index, orientation='right', count_sort=True, leaf_font_size=12, color_threshold=4)
axes[0].set_title('Sut', fontsize=16)
# plt.savefig('temp_sut_log_clusteringD4_06-11-24.png', dpi=300, bbox_inches='tight')
# plt.close()

axes[1].set_title('Wild-type', fontsize=16)
dend_wt = hierarchy.dendrogram(logWT_links, ax=axes[1], labels=all_metabDF.index, orientation='left', count_sort=True, leaf_font_size=12, color_threshold=4)
# plt.savefig('temp_wtSut_log_clusteringD4_06-11-24.png', dpi=300, bbox_inches='tight')
# exit()



import tanglegram

print(f'logWTlinks:\n{logWT_links}, shape: {logWT_links.shape}')
print(f'logSutlinks:\n{logSut_links}, shape: {logSut_links.shape}')

# dista matrices to df
sut_dist = pdist(logZ_sutDF, 'euclidean')
wt_dist = pdist(logZ_wtDF, 'euclidean')

sut_sqDist = squareform(sut_dist)
wt_sqDist = squareform(wt_dist)

print(f'square dists:\n{sut_sqDist}')

sutDist_df = pd.DataFrame(data=sut_sqDist, index=all_metabDF.index, columns=all_metabDF.index)
wtDist_df = pd.DataFrame(data=wt_sqDist, index=all_metabDF.index, columns=all_metabDF.index)

# get relevant metabolites only
sutDist_df = sutDist_df.loc[sut_clustMetabs,sut_clustMetabs]
wtDist_df = wtDist_df.loc[wt_clustMetabs,wt_clustMetabs]


# Generate a tanlgegram plot of these
fig = tanglegram.plot(sutDist_df, wtDist_df, sort=True)
fig.savefig('tanglegram_test_sub.png')

exit()





# Identify import features (metabolites) for seperating Sut & WT samples using oPLS-DA
# pyopls package: https://github.com/BiRG/pyopls

targets = np.array([1.0,1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])     # define 1,samples array of class (sut/wt) membership
targets = targets.reshape((-1,1))
print(f'targets:\n{targets}')

oplsda = OPLS(n_components=2, scale=False)  # define an OPLSDA classifier with n_components = metabolites to idetnify the importance of each metabolite

# Cross-validate the model and return the mean Q^2 score over CVs to evaluate our model
rng = np.random.default_rng(seed=42)
idxs = rng.permutation(targets.shape[0])    # get and shuffle indexs
print(f'targets[0] shape: {targets.shape[0]}, idxs: {idxs}')
# print(f'shuffled data: {logZ_All_metabDF.iloc[idxs]}\nshuffled labels: {targets[idxs]}')

print(f'missing:')
print(np.isnan(logZ_All_metabDF).any())
print(np.isnan(targets).any())
print(f'targets dtype: {targets.dtype}')

from sklearn.utils import check_array
check = check_array(targets, dtype=np.float64, copy=True, ensure_2d=False)
print(f'check: {check}')
# exit()
# exit()
# cv = cross_validate(oplsda, norm_metabDF.iloc[idxs], targets[idxs].transpose(), scoring='r2', return_train_score=True, error_score='raise', cv=5)
# print(f'Keys: {cv.keys()}')
# scores = [cv['train_score'], cv['test_score']]
# mean_scores = [np.mean(score) for score in scores]
# print(f'mean R^2: {mean_scores[0]}, mean Q^2: {mean_scores[1]}')
# exit()
# print(f'targets[0].transpose() shape: {targets.transpose().shape}\nnorm_metabDF.shape(): {norm_metabDF.shape}')
Z = oplsda.fit(X=logZ_All_metabDF.transpose(), Y=targets)
Z = oplsda.transform(X=logZ_All_metabDF.transpose())

# Get VIP scores 

t = oplsda.x_scores_
w = oplsda.x_weights_ # replace with x_rotations_ if needed
q = oplsda.y_loadings_ 

print(t)
print(w)
print(q)


vips = VIP(oplsda)
print(f'vips:\n{vips}')


# define method to format inputs and run PLS-DA on data to extract important features
def run_plsda(data, classes, standardize=True, get_weights=True, get_vip=True):

	"""
	data = input gene expression data (n_genes, n_GTExSamples)
	classes = GTEx samples for categories we wish to discrimenate [class_0, class_1]
	standardize = whether or not the data need to be standardized before running PLS-DA (like PCA, PLS-DA inputs should be standardized)
	get_weights = Whether or not we want to return the feature weights importance instead of the PLSDA object 
	"""

	# Select our data such that we have columns = class 0, class 1 and generate a corresponding class vector
	x = data.loc[:, classes[0] + classes[1]]

	if standardize:
		scaler = StandardScaler()
		x = scaler.fit_transform(x.T)   # n_samples, n_features

	y = np.array([[1]*len(classes[0]) + [0]*len(classes[1]),    # create dummy 2,n array for membership in binary classes across samples
	[0]*len(classes[0]) + [1]*len(classes[1])]
	)
	print(f'input array x:\n{x}\n x_shape = {x.shape}')
	print(f'target labels array:\n{y}\ny_shape = {y.shape}')

	# Generate a model PLS-DA
	pls = PLSRegression(scale=False, n_components=2)

	# Cross-validate the model and return the mean Q^2 score over CVs to evaluate our model
	rng = np.random.default_rng(seed=42)
	idxs = rng.permutation(y[0].shape[0])   # get and shuffle indexs

	cv = cross_validate(pls, x[idxs], y[:,idxs].transpose(), scoring='r2', return_train_score=True, cv=5)
	print(f'Keys: {cv.keys()}')
	scores = [cv['train_score'], cv['test_score']]

	# Fit model to all of our data
	pls.fit(X=x, Y=y.transpose())

	if get_weights and get_vip:
		weights = pls.x_weights_
		vips = VIP(pls)
		print(f'vip shape: {vips.shape}')
		vips = vips.reshape(-1,1)
		print(f'new shape: {vips.shape}')
		return weights, scores, vips
	elif get_weights:
		return pls.x_weights_, scores

	elif get_vip:
		vips = VIP(pls)
		print(f'vip shape:\n{vips}')
		return scores, vips

	else:
		return pls.transform(X=x, Y=y.tranpose()), scores

