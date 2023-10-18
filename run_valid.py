import matplotlib.pyplot as plt

# from common import *
from sklearn import metrics


#################################################################################################


def np_binary_cross_entropy_loss(probability, truth):
    probability = probability.astype(np.float64)
    probability = np.nan_to_num(probability, nan=1, posinf=1, neginf=0)

    p = np.clip(probability, 1e-5, 1 - 1e-5)
    y = truth

    loss = -y * np.log(p) - (1 - y) * np.log(1 - p)
    loss = loss.mean()
    return loss

def get_f1score(probability, truth, threshold = np.linspace(0, 1, 50)):
    f1score = []
    precision=[]
    recall=[]
    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict >= 0.5) & (truth >= 0.5)).sum()
        fp = ((predict >= 0.5) & (truth < 0.5)).sum()
        fn = ((predict < 0.5) & (truth >= 0.5)).sum()

        r = tp / (tp + fn + 1e-3)
        p = tp / (tp + fp + 1e-3)
        f1 = 2 * r * p / (r + p + 1e-3)
        f1score.append(f1)
        precision.append(p)
        recall.append(r)
    f1score = np.array(f1score)
    precision = np.array(precision)
    recall = np.array(recall)
    return f1score, precision, recall, threshold

def plot_auc(cancer_p, cancer_t, figure_num):
    plt.figure(figure_num)
    spacing=50
    cancer_t = cancer_t.astype(int)
    pos, bin = np.histogram(cancer_p[cancer_t == 1], np.linspace(0, 1, spacing))
    neg, bin = np.histogram(cancer_p[cancer_t == 0], np.linspace(0, 1, spacing))
    pos = pos / (cancer_t == 1).sum()
    neg = neg / (cancer_t == 0).sum()
    #print(pos)
    #print(neg)
    # plt.plot(bin[1:],neg, alpha=1)
    # plt.plot(bin[1:],pos, alpha=1)
    bin = (bin[1:] + bin[:-1]) / 2
    plt.bar(bin, neg, width=1/spacing, label='neg', alpha=0.5)
    plt.bar(bin, pos, width=1/spacing, label='pos', alpha=0.5)
    plt.legend()
    #plt.yscale('log')
    # if is_show:
    #     plt.show()
    # return  plt.gcf()

def compute_metric(cancer_p, cancer_t):

    fpr, tpr, thresholds = metrics.roc_curve(cancer_t, cancer_p)
    auc = metrics.auc(fpr, tpr)

    f1score, precision, recall, threshold = get_f1score(cancer_p, cancer_t)
    i = f1score.argmax()
    f1score, precision, recall, threshold = f1score[i], precision[i], recall[i], threshold[i]

    specificity = ((cancer_p < threshold ) & ((cancer_t <= 0.5))).sum() / (cancer_t <= 0.5).sum()
    sensitivity = ((cancer_p >= threshold) & ((cancer_t >= 0.5))).sum() / (cancer_t >= 0.5).sum()

    return {
        'auc': auc,
        'threshold': threshold,
        'f1score': f1score,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }

def compute_pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
            #cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp+1e-8)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def print_all_metric(valid_df):

	print(f'{"    ": <16}    \tauc      @th     f1      | 	prec    recall  | 	sens    spec ')
	#log.write(f'{"    ": <16}    \t0.77902	0.44898	0.28654 | 	0.32461	0.25726 | 	0.25726	0.98794\n')
	for site_id in [0,1,2]:

		#log.write(f'*** site_id [{site_id}] ***\n')
		#log.write(f'\n')

		if site_id>0:
			site_df = valid_df[valid_df.site_id == site_id].reset_index(drop=True)
		else:
			site_df = valid_df
		# ---

		gb = site_df
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"single image": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)


		# ---

		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"grouby mean()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)

		# ---
		gb = site_df[['patient_id', 'laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).max()
		m = compute_metric(gb.cancer_p, gb.cancer_t)
		text = f'{"grouby max()": <16} [{site_id}]'
		text += f'\t{m["auc"]:0.5f}'
		text += f'\t{m["threshold"]:0.5f}'
		text += f'\t{m["f1score"]:0.5f} | '
		text += f'\t{m["precision"]:0.5f}'
		text += f'\t{m["recall"]:0.5f} | '
		text += f'\t{m["sensitivity"]:0.5f}'
		text += f'\t{m["specificity"]:0.5f}'
		#text += '\n'
		print(text)
		print(f'--------------\n')


# main #################################################################
if __name__ == '__main__':
    # run_valid()
    #run_more()
    #exit(0)


    # '00059990'
    # valid_df = pd.read_csv(f'{root_dir}/result/run300/kaggle/nextvit-b-1536-gpu-aug0-01/fold-0/valid/swa/valid_df.csv')
    # cancer_t = np.load(f'{root_dir}/result/run300/kaggle/nextvit-b-1536-gpu-aug0-01/fold-0/valid/swa/cancer_t.npy', )
    # cancer_p = np.load(f'{root_dir}/result/run300/kaggle/nextvit-b-1536-gpu-aug0-01/fold-0/valid/swa/cancer_p.npy', )



    # valid_df1 = pd.read_csv(f'{root_dir}/result/run300/kaggle/effb4-1536-baseline-gpu-aug0-01/fold-0/valid/swa/valid_df.csv')
    # cancer_t1 = np.load(f'{root_dir}/result/run300/kaggle/effb4-1536-baseline-gpu-aug0-01/fold-0/valid/swa/cancer_t.npy', )
    # cancer_p1 = np.load(f'{root_dir}/result/run300/kaggle/effb4-1536-baseline-gpu-aug0-01/fold-0/valid/swa/cancer_p.npy', )
    # cancer_p = (cancer_p**0.5 + cancer_p1**0.5)/2


    # valid_df = pd.read_csv(f'{root_dir}/result/run300/kaggle/nextvit-s-2048-gpu-aug0-01/fold-0/valid/swa1/valid_df.csv')
    # cancer_t = np.load(f'{root_dir}/result/run300/kaggle/nextvit-s-2048-gpu-aug0-01/fold-0/valid/swa1/cancer_t.npy', )
    # cancer_p = np.load(f'{root_dir}/result/run300/kaggle/nextvit-s-2048-gpu-aug0-01/fold-0/valid/swa1/cancer_p.npy', )
    #
    #
    # valid_df = pd.read_csv(f'{root_dir}/result/run300/kaggle/effb0-1024-gpu-aug0-01-exp_loss/fold-0/valid/swa/valid_df.csv')
    # cancer_t =     np.load(f'{root_dir}/result/run300/kaggle/effb0-1024-gpu-aug0-01-exp_loss/fold-0/valid/swa/cancer_t.npy', )
    # cancer_p =     np.load(f'{root_dir}/result/run300/kaggle/effb0-1024-gpu-aug0-01-exp_loss/fold-0/valid/swa/cancer_p.npy', )


    # valid_df = pd.read_csv(f'{root_dir}/result/run300/kaggle/gfnet_h_s-1536-gpu-aug0-04/fold-0/valid/00024488/valid_df.csv')
    # cancer_t =     np.load(f'{root_dir}/result/run300/kaggle/gfnet_h_s-1536-gpu-aug0-04/fold-0/valid/00024488/cancer_t.npy', )
    # cancer_p =     np.load(f'{root_dir}/result/run300/kaggle/gfnet_h_s-1536-gpu-aug0-04/fold-0/valid/00024488/cancer_p.npy', )

    # experiment='run05/efficientnet_b2-2048-aug00-01'
    # iteration='swa'
    # valid_df = pd.read_csv(f'{root_dir}/result/{experiment}/fold-0/valid/{iteration}/valid_df.csv')
    # #cancer_t =     np.load(f'{root_dir}/result/{experiment}/fold-0/valid/{iteration}/cancer_t.npy', )
    # cancer_p =     np.load(f'{root_dir}/result/{experiment}/fold-0/valid/{iteration}/cancer_p.npy', )
    # cancer_t = valid_df.cancer.values


    experiment='output/train_crop_voilut_1024'
    root_dir = 'output/train_crop_voilut_1024'
    iteration='swa'
    valid_df = pd.read_csv('kfold_train.csv')
    valid_df= valid_df[valid_df.kfold==0]
    cancer_t =     np.load(f'{root_dir}/checkpoint-fold-0_true_val.npy',)
    cancer_p =     np.load(f'{root_dir}/checkpoint-fold-0_ddp_ep_1_predictions.npy', )
    #cancer_p =     np.load('/home/titanx/hengck/share1/kaggle/2022/rsna-breast-mammography/code/dummy-01/[notebook]/nextvit-b-1536-faster1/probability.npy')

    #--------------------------

    valid_df.loc[:, 'cancer_p'] = cancer_p #**0.5 #(cancer_p1 +cancer_p)/2 #cancer_p#
    valid_df.loc[:, 'cancer_t'] = cancer_t
    print_all_metric(valid_df)


    gb = valid_df[['site_id', 'patient_id','laterality', 'cancer_t', 'cancer_p']].groupby(['patient_id', 'laterality']).mean()
    gb.loc[:, 'cancer_t'] = gb.cancer_t.astype(int)
    m = compute_metric(gb.cancer_p, gb.cancer_t)
    text = f'{"grouby mean()": <16}'
    text += f'\t{m["auc"]:0.5f}'
    text += f'\t{m["threshold"]:0.5f}'
    text += f'\t{m["f1score"]:0.5f} | '
    text += f'\t{m["precision"]:0.5f}'
    text += f'\t{m["recall"]:0.5f} | '
    text += f'\t{m["sensitivity"]:0.5f}'
    text += f'\t{m["specificity"]:0.5f}'
    text += '\n'
    print(text)

    pfbeta = compute_pfbeta(gb.cancer_t.values, gb.cancer_p.values, beta=1)
    print('pfbeta',pfbeta)
    plot_auc(gb.cancer_p, gb.cancer_t, figure_num=100)


    f1score, precision, recall, threshold = get_f1score(gb.cancer_p, gb.cancer_t)
    i = f1score.argmax()
    f1score_max, precision_max, recall_max, threshold_max = f1score[i], precision[i], recall[i], threshold[i]
    print(f1score_max, precision_max, recall_max, threshold_max)

    precision, recall, threshold = metrics.precision_recall_curve(gb.cancer_t, gb.cancer_p)
    auc = metrics.auc(recall, precision)

    _, ax = plt.subplots(figsize=(5, 5))

    f_scores = [0.2,0.3,0.4,0.5,0.6,0.7,0.8] #np.linspace(0.2, 0.8, num=8)
    print(f_scores)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    ax.plot([0,1],[0,1], color="gray", alpha=0.2)
    ax.plot(recall,precision)
    s=ax.scatter(recall[:-1],precision[:-1],c=threshold,cmap='hsv')
    ax.scatter(recall_max, precision_max,s=30,c='k')

    #---
    precision, recall, threshold = metrics.precision_recall_curve(gb.cancer_t[gb.site_id==1], gb.cancer_p[gb.site_id==1])
    ax.plot(recall,precision, '--', label='site_id=1')
    precision, recall, threshold = metrics.precision_recall_curve(gb.cancer_t[gb.site_id==2], gb.cancer_p[gb.site_id==2])
    ax.plot(recall,precision, '--', label='site_id=2')




    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    text=''
    text+=f'MAX f1score {f1score_max: 0.5f} @ th = {threshold_max: 0.5f}\n'
    text+=f'prec {precision_max: 0.5f}, recall {recall_max: 0.5f}, pr-auc {auc: 0.5f}\n'

    plt.legend()
    plt.title(text)
    plt.colorbar(s,label='threshold')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()



