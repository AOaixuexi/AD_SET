from .basic_metrics import basic_metricor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches 

def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None):
    grader = basic_metricor()
    
    R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True) #
    
    L, fpr, tpr= grader.metric_new(label, score, plot_ROC=True)
    precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)
    # print(range_anomaly)
    
    # max_length = min(len(score),len(data), 20000)
    max_length = len(score)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    
    f3_ax1 = fig3.add_subplot(gs[0, :-1])
    plt.tick_params(labelbottom=False)

    plt.plot(data[:max_length],'k')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
        
    # L = [auc, precision, recall, f, Rrecall, ExistenceReward, 
    #       OverlapReward, Rprecision, Rf, precision_at_k]
    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    # plt.tick_params(labelbottom=False)
    L1 = [ '%.2f' % elem for elem in L]
    plt.plot(score[:max_length])
    plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    plt.ylabel('score')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    #plot the data
    f3_ax3 = fig3.add_subplot(gs[2, :-1])
    index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    f3_ax4 = fig3.add_subplot(gs[0, -1])
    plt.plot(fpr, tpr)
    # plt.plot(R_fpr,R_tpr)
    # plt.title('R_AUC='+str(round(R_AUC,3)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.legend(['ROC','Range-ROC'])
    
    # f3_ax5 = fig3.add_subplot(gs[1, -1])
    # plt.plot(recall, precision)
    # plt.plot(R_tpr[:-1],R_prec)   # I add (1,1) to (TPR, FPR) at the end !!!
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(['PR','Range-PR'])

    # print('AUC=', L1[0])
    # print('F=', L1[3])

    plt.suptitle(fileName + '    window='+str(slidingWindow) +'   '+ modelName
    +'\nAUC='+L1[0]+'     R_AUC='+str(round(R_AUC,2))+'     Precision='+L1[1]+ '     Recall='+L1[2]+'     F='+L1[3]
    + '     ExistenceReward='+L1[5]+'   OverlapReward='+L1[6]
    +'\nAP='+str(round(AP,2))+'     R_AP='+str(round(R_AP,2))+'     Precision@k='+L1[9]+'     Rprecision='+L1[7] + '     Rrecall='+L1[4] +'    Rf='+L1[8]
    )
    
def printResult(data, label, score, slidingWindow, fileName, modelName):
    grader = basic_metricor()
    R_AUC = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=False) #
    L= grader.metric_new(label, score, plot_ROC=False)
    L.append(R_AUC)
    return L



import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from TSB_AD.evaluation.basic_metrics import basic_metricor


def plot_func_ts(data, label, file_name):

    grader = basic_metricor()
    range_anomaly = grader.range_convers_new(label)

    # Create the main trace
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        line=dict(color='#0248b5', width=2),      # [#0b5920, 0248b5]
        name='Time Series'
    ))

    # Highlight anomalies
    for r in range_anomaly:
        if r[0] == r[1]:
            fig.add_trace(go.Scatter(
                x=[r[0]],
                y=[data[r[0]]],
                mode='markers',
                marker=dict(color='red', size=5),
                name='Anomaly Point'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(r[0], r[1]+1)),
                y=data[r[0]:r[1]+1],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Anomaly Range {r[0]}-{r[1]}'
            ))
    fig.update_layout(title=file_name, height=300, plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(
        tickfont=dict(size=18),
        showline=True,
        linecolor='black',
        linewidth=2
        ))
    return fig

def plot_func_score(label, anomaly_score, file_name):
    auc_roc = roc_auc_score(label, anomaly_score)
    auc_pr = average_precision_score(label, anomaly_score)

    thresholds = np.linspace(anomaly_score.min(), anomaly_score.max(), 100)
    f1_scores = []

    for threshold in thresholds:
        predictions = (anomaly_score > threshold).astype(int)
        f1 = f1_score(label, predictions)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = max(f1_scores)

    predictions = (anomaly_score > best_threshold).astype(int)

    false_negatives = np.where((label == 1) & (predictions == 0))[0]
    false_positives = np.where((label == 0) & (predictions == 1))[0]
    true_positives = np.where((label == 1) & (predictions == 1))[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(anomaly_score))),
        y=anomaly_score,
        mode='lines',
        line=dict(color='#26a9b5', width=1.5),
        name='Anomaly Score'
    ))

    fig.add_trace(go.Scatter(
        x=[0, len(anomaly_score) - 1],
        y=[best_threshold, best_threshold],
        mode='lines',
        line=dict(color='red', width=1, dash='dash'),
        name='Threshold'
    ))

    fig.add_trace(go.Scatter(
        x=false_negatives,
        y=anomaly_score[false_negatives],
        mode='markers',
        marker=dict(color='orange', size=5, symbol='x'),
        name='False Negatives'
    ))

    fig.add_trace(go.Scatter(
        x=false_positives,
        y=anomaly_score[false_positives],
        mode='markers',
        marker=dict(color='purple', size=5, symbol='circle'),
        name='False Positives'
    ))

    fig.add_trace(go.Scatter(
        x=true_positives,
        y=anomaly_score[true_positives],
        mode='markers',
        marker=dict(color='green', size=5, symbol='star'),
        name='True Positives'
    ))

    fig.update_layout(
        title=f"{file_name}<br>AUC-ROC: {auc_roc:.2f}, AUC-PR: {auc_pr:.2f}, Best F1: {best_f1_score:.2f}",
        height=300,
        xaxis=dict(
            tickfont=dict(size=18),
            title='Time',
            showline=True,
            linecolor='black',
            linewidth=2
        ),
        yaxis=dict(
            title='Anomaly Score',
            showline=True,
            linecolor='black',
            linewidth=2
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig