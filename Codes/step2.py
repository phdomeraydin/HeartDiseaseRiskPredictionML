"""Step 2: nested CV, repeated CV and final internal holdout for eight models.

Run after step1.py: python step2.py
Progress is saved after EACH outer fold/model; restarting skips completed work.
All optimization, imputation and synthetic training augmentation stay inside
their respective training folds. Grid sizes are deliberately smaller than V10.
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

import step1 as core


def settings_and_specs():
    specs = core.model_specs()
    grids = {name:grid for name,(_,grid) in specs.items()}
    settings = {"Dataset_SHA256":core.data_fingerprint(), "Seed":core.SEED,
                "Holdout_fraction":core.SPLIT_FRACTION,
                "Nested_outer_folds":10,"Nested_inner_folds":5,
                "Repeated_outer_folds":5,"Repeated_repeats":3,
                "Repeated_inner_folds":3,"Augmentation_ratio":.5,
                "Nested_search_spaces":grids,
                "Repeated_search_spaces":core.compact_repeated_grids(),
                "CIs":"fold-based intervals describe resampling variability; not independent patient-population confidence intervals",
                "Pairwise_tests":"exploratory: outer-fold scores share training data"}
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True,default=str).encode()
                          + open(core.__file__,"rb").read()
                          + open(__file__,"rb").read()).hexdigest()[:16]
    settings["Run_ID"]=digest
    path=core.RESULTS/"step2_settings.json"
    if path.exists():
        saved=json.loads(path.read_text(encoding="utf-8"))
        if saved["Run_ID"]!=digest:
            raise ValueError("Saved step-2 results use different data, grids or code. Move the old results directory before rerunning.")
    else:
        path.write_text(json.dumps(settings,indent=2,default=str),encoding="utf-8")
    return settings,specs


def score(estimator,X,y):
    prediction=estimator.predict(X)
    probability=(estimator.predict_proba(X)[:,1]
                 if hasattr(estimator,"predict_proba") else None)
    values=probability if probability is not None else estimator.decision_function(X)
    row={"N":len(y),"Accuracy":accuracy_score(y,prediction),
         "Precision":precision_score(y,prediction,zero_division=0),
         "Recall":recall_score(y,prediction,zero_division=0),
         "F1":f1_score(y,prediction,zero_division=0),
         "ROC_AUC":roc_auc_score(y,values),
         "Brier":brier_score_loss(y,probability) if probability is not None else np.nan}
    return row, prediction, values, probability


def tuned(train_X,train_y,model,grid,inner_folds,ratio=.5,noise=1,encoding="onehot"):
    estimator=core.ClinicalEstimator(model=model,ratio=ratio,
                                     noise_multiplier=noise,encoding=encoding)
    search=GridSearchCV(estimator,grid,scoring="roc_auc",
                        cv=StratifiedKFold(inner_folds,shuffle=True,random_state=core.SEED),
                        n_jobs=1,refit=True,error_score="raise")
    search.fit(train_X,train_y)
    return search.best_estimator_,search.best_params_


def progress(name,run_id):
    path=core.RESULTS/name
    if not path.exists():return []
    rows=pd.read_excel(path).to_dict("records")
    if any(str(r["Run_ID"])!=run_id for r in rows):
        raise ValueError(f"Run identifier mismatch in {name}; old results cannot be reused")
    return rows


def mean_intervals(rows,name,fold_name):
    result=[]
    for model,group in pd.DataFrame(rows).groupby("Model",sort=False):
        record={"Model":model,"N_outer_scores":len(group),
                "CI_interpretation":"descriptive fold-resampling interval; correlated folds"}
        for field in ("Accuracy","Precision","Recall","F1","ROC_AUC"):
            series=group[field].to_numpy(dtype=float)
            mean=float(np.mean(series));sd=float(np.std(series,ddof=1))
            low,high=stats.t.interval(.95,len(series)-1,loc=mean,scale=sd/np.sqrt(len(series)))
            record.update({field+"_mean":mean,field+"_sd":sd,
                           field+"_CI_lower":float(low),field+"_CI_upper":float(high)})
        result.append(record)
    core.save_table(pd.DataFrame(result),name)


def cv_phase(label,filename,finalname,splitter,inner_folds,grids,specs,X,y,run_id,total):
    rows=progress(filename,run_id)
    seen={(str(row["Model"]),int(row["Outer_Fold"])) for row in rows}
    partitions=list(splitter.split(X,y))
    for model_name,(model,_) in specs.items():
        for fold,(train,val) in enumerate(partitions,1):
            if (model_name,fold) in seen:continue
            start=time.monotonic()
            print(f"{label}: {model_name}, fold {fold}/{total} START",flush=True)
            best,params=tuned(X.iloc[train],y.iloc[train],model,grids[model_name],inner_folds)
            metrics,_,_,_=score(best,X.iloc[val],y.iloc[val])
            rows.append({"Run_ID":run_id,"Model":model_name,"Outer_Fold":fold,
                         "Best_Params":json.dumps(params,sort_keys=True),**metrics})
            core.save_table(pd.DataFrame(rows),filename)
            print(f"{label}: {model_name} fold {fold} DONE; ROC-AUC={metrics['ROC_AUC']:.4f}; elapsed={time.monotonic()-start:.1f}s",flush=True)
    if len(rows)!=len(specs)*total:raise RuntimeError(f"{label} incomplete; restart step2.py")
    core.save_table(pd.DataFrame(rows),finalname)
    mean_intervals(rows,finalname.replace("fold_level_results","results_with_95CI").replace("repeated_kfold_results","repeated_summary_with_95CI"),"Outer_Fold")
    return rows


def final_holdout(specs,X_dev,y_dev,X_test,y_test,run_id):
    filename="final_holdout_test_results_PROGRESS.xlsx"
    rows=progress(filename,run_id)
    done={str(row["Model"]) for row in rows}
    preds=progress("final_holdout_predictions_PROGRESS.xlsx",run_id)
    pred_done={str(row["Model"]) for row in preds}
    train_rows=progress("train_test_accuracy_PROGRESS.xlsx",run_id)
    train_done={str(row["Model"]) for row in train_rows}
    if not done.issubset(pred_done) or not done.issubset(train_done):
        raise ValueError("A completed holdout metric lacks its predictions or training checkpoint")
    # If execution stopped between checkpoint writes, discard only that
    # unfinished model's auxiliary rows and recompute the whole model.
    if done!=pred_done or done!=train_done:
        preds=[row for row in preds if str(row["Model"]) in done]
        train_rows=[row for row in train_rows if str(row["Model"]) in done]
        if preds:core.save_table(pd.DataFrame(preds),"final_holdout_predictions_PROGRESS.xlsx")
        if train_rows:core.save_table(pd.DataFrame(train_rows),"train_test_accuracy_PROGRESS.xlsx")
    for name,(model,grid) in specs.items():
        if name in done:continue
        print(f"Internal holdout: fitting {name}",flush=True)
        fitted,params=tuned(X_dev,y_dev,model,grid,5)
        measures,prediction,scores,proba=score(fitted,X_test,y_test)
        train_measures,_,_,_=score(fitted,X_dev,y_dev)
        params_json=json.dumps(params,sort_keys=True)
        new_preds=[{"Run_ID":run_id,"Model":name,"Input_row_1_based":int(idx)+1,
                    "Target":int(truth),"Predicted":int(pred),"Decision_score":float(val),
                    "Probability":float(prob) if proba is not None else np.nan}
                   for idx,truth,pred,val,prob in zip(
                       X_test.index,y_test,prediction,scores,
                       proba if proba is not None else np.full(len(y_test),np.nan))]
        # The prediction table is saved FIRST; completed-model rows follow.
        core.save_table(pd.DataFrame(preds+new_preds),"final_holdout_predictions_PROGRESS.xlsx")
        core.save_table(pd.DataFrame(train_rows+[{"Run_ID":run_id,"Model":name,
                        "Train_Accuracy":train_measures["Accuracy"],
                        "Holdout_Accuracy":measures["Accuracy"]}]),"train_test_accuracy_PROGRESS.xlsx")
        core.save_table(pd.DataFrame(rows+[{"Run_ID":run_id,"Model":name,
                        "Best_Params":params_json,**measures}]),filename)
        rows.append({"Run_ID":run_id,"Model":name,"Best_Params":params_json,**measures})
        preds+=new_preds
        train_rows.append({"Run_ID":run_id,"Model":name,
                           "Train_Accuracy":train_measures["Accuracy"],
                           "Holdout_Accuracy":measures["Accuracy"]})
        print(f"Internal holdout: {name} DONE; ROC-AUC={measures['ROC_AUC']:.4f}",flush=True)
    core.save_table(pd.DataFrame(rows),"final_holdout_test_results.xlsx")
    core.save_table(pd.DataFrame(preds),"final_holdout_predictions.xlsx")
    core.save_table(pd.DataFrame(train_rows),"train_test_accuracy.xlsx")


def main():
    core.manifest()
    settings,specs=settings_and_specs()
    print("Run ID:",settings["Run_ID"],"Output:",core.RESULTS,flush=True)
    raw=core.load_data()
    data=raw.copy();data[core.COLS[:-1]]=core.cleaned_features(raw)
    X_dev,X_test,y_dev,y_test=core.split_data(data)
    run_id=settings["Run_ID"]
    cv_phase("Nested CV","nested_cv_fold_level_results_PROGRESS.xlsx",
             "nested_cv_fold_level_results.xlsx",
             StratifiedKFold(10,shuffle=True,random_state=core.SEED),5,
             {n:g for n,(_,g) in specs.items()},specs,X_dev,y_dev,run_id,10)
    cv_phase("Repeated CV","repeated_kfold_results_PROGRESS.xlsx",
             "repeated_kfold_results.xlsx",
             RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=core.SEED),3,
             core.compact_repeated_grids(),specs,X_dev,y_dev,run_id,15)
    final_holdout(specs,X_dev,y_dev,X_test,y_test,run_id)
    print("Step 2 completed. Run python step3.py",flush=True)


if __name__=="__main__":
    main()
