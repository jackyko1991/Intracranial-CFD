from re import I
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import multiprocessing

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

def plot_coefficients(coefs, n_highlight,semilogx=True):
    fig ,ax = plt.subplots(figsize=(9,6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        if semilogx:
            ax.semilogx(alphas, row[1:], ".-", label=row.Index)
        else:
            ax.plot(alphas, row[1:], ".-", label=row.Index)
    
    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef-0.03, name + "   ",
            horizontalalignment="left",
            verticalalignment="top"
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    # ax.legend(loc="best")

    return fig, ax

def main():
    csv_file = "Z:/data/intracranial/recurrent/results_recurrent.csv"
    gcv_output_file = "Z:/data/intracranial/recurrent/alpha_cox_ridge_model_recurrent.csv"
    result = pd.read_csv(csv_file)

    result_X = result[[
        #"annual followup",
        #"group(medical=0,stent=1)",
        "age",
        "degree of stenosis(%)",
        "radius min(mm)",
        "translesion peak presssure(mmHg)",
        "translesion presssure ratio",
        "translesion peak pressure gradient(mmHgmm^-1)",
        "translesion pressure gradient ratio",	
        "translesion peak velocity(ms^-1)",	
        "translesion velocity ratio",
        "translesion peak velocity gradient(ms^-1mm^-1)",
        "translesion velocity gradient ratio",	
        "translesion peak vorticity(ms^-1)",	
        "translesion vorticity ratio",	
        "translesion peak vorticity gradient(Pamm^-1)",	
        "translesion vorticity gradient ratio",	
        "translesion peak wss(Pa)",
        "translesion wss ratio",
        "translesion peak wss gradient(Pamm^-1)",	
        "translesion wss gradient ratio",
        #"AIS within one year",
        ]]

    result_Y = result[[
        "AIS within one year",
        "1st AIS date delta with 1 year censor",
        "TIA within one year",
        "1st TIA date delta with 1 year censor",
        "recurrent within one year",
        "1st recurrent date delta with 1 year censor"
        ]]
    result_Y["AIS within one year"] = result_Y["AIS within one year"].astype(bool)
    result_Y["TIA within one year"] = result_Y["TIA within one year"].astype(bool)
    result_Y["recurrent within one year"] = result_Y["recurrent within one year"].astype(bool)

    result_Y_AIS = result_Y[[
        "AIS within one year",
        "1st AIS date delta with 1 year censor",
    ]]

    # result_Y_AIS = result_Y[[
    #     "recurrent within one year",
    #     "1st recurrent date delta with 1 year censor",
    # ]]

    result_Y_AIS_array = result_Y_AIS.to_records(index=False)

    warnings.simplefilter("ignore", ConvergenceWarning)

    # multivairate survival analysis using penalized cox models with ridge regression
    print("==================== Fitting multivariate Cox model with ridge regression ====================")

    search_mag_depth = 2

    alphas = 10.**np.linspace(-1*search_mag_depth,search_mag_depth,50)
    coefficients = {}
    estimator = make_pipeline(
        StandardScaler(),
        CoxPHSurvivalAnalysis()
    )

    pbar = tqdm(alphas)
    for alpha in pbar:
        estimator.set_params(coxphsurvivalanalysis__alpha=alpha)
        estimator.fit(result_X,result_Y_AIS_array)
        key = round(alpha,5)
        coefficients[key] = estimator.named_steps['coxphsurvivalanalysis'].coef_
    
    coefficients = (pd.DataFrame
        .from_dict(coefficients)
        .rename_axis(index="feature", columns="alpha")
        .set_index(result_X.columns))

    fig_cox_rigid, ax_cox_rigid = plot_coefficients(coefficients, n_highlight=5)

    # choosing penalty strength alpha and l1 ratios
    # perform 5 fold cross validation to estimate performance with c-index for each alpha
    print("==================== Grid searching optimal alpha ====================")

    cox_ridge_pipe = make_pipeline(
        StandardScaler(),
        CoxPHSurvivalAnalysis()
    )
    param_grid = {
        'coxphsurvivalanalysis__alpha': alphas
        }

    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # cv = StratifiedKFold(5, random_state=1,shuffle=True)
    gcv = GridSearchCV(cox_ridge_pipe, param_grid, return_train_score=True,error_score=0.5,n_jobs=multiprocessing.cpu_count(), cv=cv)
    gcv.fit(result_X, result_Y_AIS_array)
    cv_results = pd.DataFrame(gcv.cv_results_)
    cv_results.to_csv(gcv_output_file)

    # vizualize the results
    print("best testing alpha: {:.4f}".format(gcv.best_params_["coxphsurvivalanalysis__alpha"]))
    alphas = cv_results.param_coxphsurvivalanalysis__alpha.map(lambda x: x)
    mean_train = cv_results.mean_train_score
    std_train = cv_results.std_train_score
    mean_test = cv_results.mean_test_score
    std_test = cv_results.std_test_score

    fig_alpha, ax_alpha = plt.subplots(figsize=(9, 6))
    ax_alpha.plot(alphas, mean_train,label="training")
    ax_alpha.fill_between(alphas, mean_train - std_train, mean_train + std_train, alpha=.15)
    ax_alpha.plot(alphas, mean_test,label="testing")
    ax_alpha.fill_between(alphas, mean_test - std_test, mean_test + std_test, alpha=.15)
    ax_alpha.set_xscale("log")
    ax_alpha.set_ylabel("concordance index")
    ax_alpha.set_xlabel("alpha")
    ax_alpha.axvline(gcv.best_params_["coxphsurvivalanalysis__alpha"], linestyle="--", c="r",
        label="best testing alpha = {:.4f}, c-index = {:.4f}".format(gcv.best_params_["coxphsurvivalanalysis__alpha"],max(mean_test))
        )
    ax_alpha.axhline(0.5, color="grey", linestyle="--")
    ax_alpha.grid(True)
    ax_alpha.legend(loc="best")

    # plot of coefficients
    print("==================== Plotting Cox model regression coefficients ====================")
    best_model = gcv.best_estimator_.named_steps["coxphsurvivalanalysis"]
    best_coefs = pd.DataFrame(
        best_model.coef_,
        index=result_X.columns,
        columns=["coefficient"]
    )

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print("Number of non-zero coefficients: {}".format(non_zero))

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    fig_selected_features, ax_selected_features = plt.subplots(figsize=(10, 6))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax_selected_features, legend=False)
    ax_selected_features.set_xlabel("coefficient")
    ax_selected_features.grid(True)
    fig_selected_features.tight_layout()

    # we have get optimal alpha, now investigate how the top 5 factors affecting survival function, prediction model without kfold
    print("==================== Fitting Cox model to investigate factors affecting survival ====================")
    cph_pred = make_pipeline(
        StandardScaler(),
        CoxPHSurvivalAnalysis()
    )
    cph_pred.set_params(**gcv.best_params_)
    cph_pred.fit(result_X,result_Y_AIS_array)
    
    time_points = np.quantile(result_Y_AIS_array["1st AIS date delta with 1 year censor"], np.linspace(0, 0.6, 100))
    #time_points = np.quantile(result_Y_AIS_array["1st recurrent date delta with 1 year censor"], np.linspace(0, 0.6, 100))
    baseline_surv = cph_pred["coxphsurvivalanalysis"].baseline_survival_
    fig_feat, axs_feat = plt.subplots(2,3,figsize=(16, 8))
    
    # pick the top 6 features to analysis
    picked_feats = non_zero_coefs.loc[coef_order]["coefficient"][::-1][:6]
    std_ranges = np.arange(-5,6,2)[::-1]
    result_X_mean = result_X.mean()

    for index, (feat, value) in enumerate(picked_feats.items()):
        ax = axs_feat.flat[index]
        # plot baseline
        ax.title.set_text(feat)
        ax.step(time_points, baseline_surv(time_points), where="post", color="k", alpha=1.0, label="baseline")

        feature_mean = result_X.loc[:,feat].mean()
        feature_std = result_X.loc[:,feat].std()

        print(feat)
        for j, std in enumerate(std_ranges):
            result_X_sd = result_X_mean.copy()
            result_X_sd.name = "+{} sd".format(std) if std> 0 else "-{} sd".format(abs(std))
            result_X_sd[feat] = feature_mean + std*feature_std
            surv_fns = cph_pred.predict_survival_function(result_X_sd.to_frame().T)
            ax.step(time_points,surv_fns[0](time_points), where="post",color = "C{:d}".format(j), alpha =0.5, label=result_X_sd.name)

            # risk_score = np.exp(cph_pred.predict(result_X_sd.to_frame().T))[0]
            # print("std: {}, risk score: {}".format(std,risk_score))

            # # inverse survival function for median survival time
            # inverse_surv = interp1d(surv_fns[0](time_points)[::-1], time_points[::-1], bounds_error=False, assume_sorted=True)
            # # # S(t) = 0.5 <=> t = S^{-1}(0.5) = t
            # print("95 percentile survival time: {} days".format(inverse_surv(0.95)))

        ax.set_ylim(0.8,1.0)
        ax.set_xlabel("time")
        ax.set_ylabel("survival probability")
        ax.grid(True)
        ax.legend(loc="lower left")
    fig_feat.tight_layout()

    cox_model_regression_coef = pd.Series(cph_pred["coxphsurvivalanalysis"].coef_, index=result_X.columns)
    print("==================== Cox model regression coefficients for log hazard ratio ====================")
    print(cox_model_regression_coef)

    print("==================== Cox model hazard ratio ====================")
    print(np.exp(cox_model_regression_coef))

    # performance measurement with Harrell’s concordance index (time dependent AUC)
    prediction = cph_pred.predict(result_X)
    result = concordance_index_censored(result_Y["AIS within one year"], result_Y["1st AIS date delta with 1 year censor"], prediction)
    #result = concordance_index_censored(result_Y["recurrent within one year"], result_Y["1st recurrent date delta with 1 year censor"], prediction)
    print("Overall C-index: {}".format(result[0]))
    
    plt.show()

if __name__ == "__main__":
    main()