"""Step 3: model comparisons, RF sensitivity/ablation, explainability and plots.

Run directly after steps 1 and 2: python step3.py
No new source-wise validation is claimed: the input is one compiled dataset.
Exploratory tests on correlated cross-validation folds must not be interpreted
as independent-population tests of clinical superiority.
"""

import hashlib
import json
import time
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import step1 as core
import step2 as stage2


def require_step2(settings,specs):
    run_id=settings["Run_ID"]
    files={"nested_cv_fold_level_results.xlsx":10,
           "repeated_kfold_results.xlsx":15,
           "final_holdout_test_results.xlsx":1,
           "final_holdout_predictions.xlsx":None}
    result={}
    for filename,folds in files.items():
        path=core.RESULTS/filename
        if not path.is_file():
            raise FileNotFoundError(f"Complete step2.py before step3.py: {path}")
        frame=pd.read_excel(path)
        if not frame.Run_ID.astype(str).eq(run_id).all():
            raise ValueError(f"Stale result file: {filename}")
        target_n=len(specs)*folds if folds is not None else len(specs)*settings["holdout_n"] if "holdout_n" in settings else len(specs)*core.manifest()["holdout_n"]
        if len(frame)!=target_n:
            raise ValueError(f"Incomplete {filename}: expected {target_n} rows, found {len(frame)}")
        result[filename]=frame
    return result


def statistical_comparison(folds,names):
    matrix=folds.pivot(index="Outer_Fold",columns="Model",values="ROC_AUC")
    if matrix.isna().any().any() or len(matrix)!=10 or len(matrix.columns)!=8:
        raise ValueError("All ten paired outer-fold ROC-AUC values are required")
    ordered=matrix[names]
    stat,p=friedmanchisquare(*(ordered[name].to_numpy() for name in names))
    core.save_table(pd.DataFrame([{"Metric":"ROC_AUC","Models":8,"Outer_folds":10,
                "Friedman_statistic":stat,"p_value":p,
                "Inference":"exploratory; overlapping training sets do not provide independent replicates"}]),
                    "friedman_test_results.xlsx")
    pair=[]
    for a,b in combinations(names,2):
        w,p_raw=wilcoxon(ordered[a].to_numpy(),ordered[b].to_numpy(),
                         alternative="two-sided",zero_method="wilcox")
        pair.append({"Model_A":a,"Model_B":b,"Wilcoxon_stat":w,
                     "p_value":p_raw,"p_bonferroni":min(1.0,p_raw*28),
                     "significant_0.05":min(1.0,p_raw*28)<.05})
    core.save_table(pd.DataFrame(pair),"pairwise_wilcoxon_tests.xlsx")
    table=ordered.T.reset_index().rename(columns={"index":"Model",**{j:f"Fold_{j}" for j in range(1,11)}})
    core.save_table(table,"nested_cv_roc_auc_matrix.xlsx")


def holdout_uncertainty_and_plots(predictions,names):
    rng=np.random.default_rng(core.SEED)
    results=[]
    fig_cal,ax_cal=plt.subplots(figsize=(7,6))
    fig_roc,ax_roc=plt.subplots(figsize=(7,6))
    for name in names:
        group=predictions.loc[predictions.Model.eq(name)].sort_values("Input_row_1_based")
        y=group.Target.to_numpy(dtype=int)
        raw=group.Decision_score.to_numpy(dtype=float)
        values=group.Probability.to_numpy(dtype=float)
        has_probability=np.isfinite(values).all()
        auc_samples=[];brier_samples=[]
        # Stratified bootstrap: retain at least one observation of each class.
        zero=np.flatnonzero(y==0);one=np.flatnonzero(y==1)
        for _ in range(1500):
            sampled=np.concatenate((rng.choice(zero,len(zero),replace=True),
                                    rng.choice(one,len(one),replace=True)))
            auc_samples.append(roc_auc_score(y[sampled],raw[sampled]))
            if has_probability:
                brier_samples.append(np.mean((values[sampled]-y[sampled])**2))
        row={"Model":name,"ROC_AUC":roc_auc_score(y,raw),
             "AUC_bootstrap_95_lower":np.quantile(auc_samples,.025),
             "AUC_bootstrap_95_upper":np.quantile(auc_samples,.975),
             "Bootstrap_replicates":1500,
             "Interpretation":"internal holdout; fixed fitted model; no external validation"}
        if has_probability:
            row.update({"Brier":np.mean((values-y)**2),
                        "Brier_bootstrap_95_lower":np.quantile(brier_samples,.025),
                        "Brier_bootstrap_95_upper":np.quantile(brier_samples,.975)})
            observed,estimated=calibration_curve(y,values,n_bins=8,strategy="quantile")
            ax_cal.plot(estimated,observed,marker="o",label=name)
        results.append(row)
        fpr,tpr,_=roc_curve(y,raw)
        ax_roc.plot(fpr,tpr,label=f"{name} ({row['ROC_AUC']:.3f})")
    ax_cal.plot([0,1],[0,1],"k--",label="Perfect calibration")
    ax_cal.set(xlabel="Mean predicted probability",ylabel="Observed positive fraction",
               title="Calibration on internal holdout",xlim=(0,1),ylim=(0,1))
    ax_cal.legend(fontsize=7);fig_cal.tight_layout()
    fig_cal.savefig(core.RESULTS/"calibration_curves.png",dpi=200);plt.close(fig_cal)
    ax_roc.plot([0,1],[0,1],"k--",alpha=.5)
    ax_roc.set(xlabel="False positive rate",ylabel="True positive rate",
               title="ROC curves on internal holdout")
    ax_roc.legend(fontsize=7);fig_roc.tight_layout()
    fig_roc.savefig(core.RESULTS/"holdout_roc_curves.png",dpi=200);plt.close(fig_roc)
    core.save_table(pd.DataFrame(results),"holdout_bootstrap_95CI.xlsx")


def rf_analysis(X_dev,y_dev,X_test,y_test,holdout):
    record=holdout.loc[holdout.Model.eq("Random Forest")]
    if len(record)!=1:raise ValueError("A single completed Random Forest holdout row is required")
    params=json.loads(record.iloc[0].Best_Params)
    fitted=core.ClinicalEstimator(model=RandomForestClassifier(
        random_state=core.SEED,n_jobs=1),ratio=.5)
    fitted.set_params(**params)
    fitted.fit(X_dev,y_dev) # all parameters fixed from development-only tuning
    permutation=permutation_importance(fitted,X_test,y_test,scoring="roc_auc",
                                        n_repeats=30,random_state=core.SEED,n_jobs=1)
    perm=pd.DataFrame({"Feature":core.COLS[:-1],
                       "ROC_AUC_decrease_mean":permutation.importances_mean,
                       "ROC_AUC_decrease_sd":permutation.importances_std})
    core.save_table(perm.sort_values("ROC_AUC_decrease_mean",ascending=False),
                    "random_forest_permutation_importance.xlsx")
    encoded=fitted.preprocessor_.get_feature_names_out()
    importance=fitted.model_.feature_importances_
    if len(encoded)!=len(importance):raise AssertionError("Feature-name alignment failed")
    names=[]
    for col in encoded:
        source=next((x for x in core.NOMINAL if col.startswith(x+"_")),col)
        names.append(source)
    encoded_table=pd.DataFrame({"Encoded_feature":encoded,"Original_feature":names,
                                "Impurity_importance":importance})
    core.save_table(encoded_table,"random_forest_encoded_feature_importance.xlsx")
    grouped=encoded_table.groupby("Original_feature",as_index=False).Impurity_importance.sum()
    grouped=grouped.rename(columns={"Original_feature":"Feature"})
    core.save_table(grouped.sort_values("Impurity_importance",ascending=False),
                    "random_forest_feature_importance_aggregated.xlsx")
    merged=grouped.merge(perm,on="Feature",validate="one_to_one")
    core.save_table(merged.sort_values("ROC_AUC_decrease_mean",ascending=False),
                    "feature_importance_comparison.xlsx")
    fig,ax=plt.subplots(figsize=(8,5))
    plot=perm.sort_values("ROC_AUC_decrease_mean")
    ax.barh(plot.Feature,plot.ROC_AUC_decrease_mean,xerr=plot.ROC_AUC_decrease_sd)
    ax.set(xlabel="Decrease in internal holdout ROC-AUC",title="Random Forest permutation importance")
    fig.tight_layout();fig.savefig(core.RESULTS/"rf_permutation_importance.png",dpi=200);plt.close(fig)


def rf_sensitivity(X,y,run_id,step3_id):
    rf,full=core.model_specs()["Random Forest"]
    # A compact, prespecified grid used identically for all tuned configurations.
    rf_grid={"model__n_estimators":[150],"model__max_depth":[5,None],
             "model__min_samples_leaf":[1,4]}
    basic=RandomForestClassifier(n_estimators=150,max_depth=None,
                                  min_samples_leaf=1,random_state=core.SEED,n_jobs=1)
    configs=[("Baseline integer, no augmentation or tuning","component","integer",0,1,False),
             ("One-hot, no augmentation or tuning","component","onehot",0,1,False),
             ("One-hot, 50% augmentation, no tuning","component","onehot",.5,1,False),
             ("One-hot, tuning, no augmentation","component","onehot",0,1,True),
             ("One-hot, 50% augmentation and tuning","component","onehot",.5,1,True),
             ("Integer, 50% augmentation and tuning","encoding","integer",.5,1,True),
             ("25% augmentation","sensitivity","onehot",.25,1,True),
             ("100% augmentation","sensitivity","onehot",1,1,True),
             ("50% augmentation, 0.5 noise","sensitivity","onehot",.5,.5,True),
             ("50% augmentation, 1.5 noise","sensitivity","onehot",.5,1.5,True)]
    path=core.RESULTS/"rf_ablation_fold_scores_PROGRESS.xlsx"
    rows=pd.read_excel(path).to_dict("records") if path.exists() else []
    if any(str(row["Run_ID"])!=run_id or str(row["Step3_ID"])!=step3_id for row in rows):
        raise ValueError("Old RF ablation checkpoints belong to a different code/data version")
    seen={(str(r["Configuration"]),int(r["Fold"])) for r in rows}
    partitions=list(StratifiedKFold(5,shuffle=True,random_state=core.SEED).split(X,y))
    for label,kind,encoding,ratio,noise,tune in configs:
        for fold,(training,validation) in enumerate(partitions,1):
            if (label,fold) in seen:continue
            start=time.monotonic()
            print(f"RF analysis: {label}, fold {fold}/5 START",flush=True)
            if tune:
                model,_=stage2.tuned(X.iloc[training],y.iloc[training],rf,
                                     rf_grid,3,ratio,noise,encoding)
            else:
                model=core.ClinicalEstimator(model=basic,ratio=ratio,
                                             noise_multiplier=noise,encoding=encoding)
                model.fit(X.iloc[training],y.iloc[training])
            metrics,_,_,_=stage2.score(model,X.iloc[validation],y.iloc[validation])
            rows.append({"Run_ID":run_id,"Step3_ID":step3_id,
                         "Configuration":label,"Analysis":kind,"Encoding":encoding,
                         "Augmentation_ratio":ratio,"Noise_multiplier":noise,
                         "Tuning":tune,"Fold":fold,**metrics})
            core.save_table(pd.DataFrame(rows),"rf_ablation_fold_scores_PROGRESS.xlsx")
            print(f"RF analysis: fold {fold} DONE; AUC={metrics['ROC_AUC']:.4f}; {time.monotonic()-start:.1f}s",flush=True)
    full_frame=pd.DataFrame(rows)
    core.save_table(full_frame,"rf_ablation_fold_scores.xlsx")
    summary=full_frame.groupby(["Configuration","Analysis","Encoding",
        "Augmentation_ratio","Noise_multiplier","Tuning"],dropna=False).agg(
        Mean_ROC_AUC=("ROC_AUC","mean"),SD_ROC_AUC=("ROC_AUC","std"),
        Mean_Brier=("Brier","mean"),N_folds=("Fold","count")).reset_index()
    core.save_table(summary[summary.Analysis.eq("component")],"component_ablation.xlsx")
    ref=summary[summary.Configuration.eq("One-hot, 50% augmentation and tuning")]
    zero=summary[summary.Configuration.eq("One-hot, tuning, no augmentation")]
    core.save_table(pd.concat([zero,ref,summary[summary.Analysis.eq("sensitivity")]],ignore_index=True),
                    "augmentation_sensitivity.xlsx")
    core.save_table(summary[summary.Configuration.isin(("Integer, 50% augmentation and tuning",
                       "One-hot, 50% augmentation and tuning"))],"encoding_comparison.xlsx")


def main():
    core.manifest()
    settings,specs=stage2.settings_and_specs()
    outputs=require_step2(settings,specs)
    run_id=settings["Run_ID"]
    step3_id=hashlib.sha256((run_id+open(__file__,encoding="utf-8").read()).encode()).hexdigest()[:16]
    raw=core.load_data();data=raw.copy()
    data[core.COLS[:-1]]=core.cleaned_features(raw)
    X_dev,X_test,y_dev,y_test=core.split_data(data)
    names=list(specs)
    statistical_comparison(outputs["nested_cv_fold_level_results.xlsx"],names)
    holdout_uncertainty_and_plots(outputs["final_holdout_predictions.xlsx"],names)
    rf_analysis(X_dev,y_dev,X_test,y_test,outputs["final_holdout_test_results.xlsx"])
    rf_sensitivity(X_dev,y_dev,run_id,step3_id)
    print("Step 3 completed. Results:",core.RESULTS,flush=True)


if __name__=="__main__":
    main()
