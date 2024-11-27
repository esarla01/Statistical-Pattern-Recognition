'''
Summary
-------
Plot predicted mean + high confidence interval for PPE estimator
across different orders of the polynomial features
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("notebook")

import regr_viz_utils
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator
from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator


if __name__ == '__main__':    
    x_train_ND, t_train_N, x_test_ND, t_test_N = regr_viz_utils.load_dataset()

    # Polynomial orders to try
    order_list = [1, 4, 10]

    # Set precisions of prior (alpha) and likelihood (beta)
    alpha_list = 0.0001 * np.ones(3)
    beta_list = [0.1, 1.0, 10]

    # Set training set size
    N = 8 

    ppe_fig, ppe_axgrid = regr_viz_utils.prepare_x_vs_t_fig(order_list)
    xgrid_G1 = regr_viz_utils.prepare_xgrid_G1(x_train_ND)

    # Loop over order of polynomial features
    for fig_col_id in range(len(order_list)):
        order = order_list[fig_col_id]
        alpha = alpha_list[fig_col_id]
        beta = beta_list[fig_col_id]
        cur_ppe_ax = ppe_axgrid[0, fig_col_id]

        feature_transformer = PolynomialFeatureTransform(
            order=order, input_dim=1)

        # Train PPE estimator using only first N examples
        ppe_estimator = LinearRegressionPosteriorPredictiveEstimator(
                feature_transformer, alpha=alpha, beta=beta)
        ppe_estimator.fit(x_train_ND[:N], t_train_N[:N])

        # Compute score on train and test
        ppe_tr_score = ppe_estimator.score(x_train_ND[:N], t_train_N[:N])
        ppe_te_score = ppe_estimator.score(x_test_ND, t_test_N)
        print("order %2d  %8.4f tr score  %8.4f te score" % (
            order, ppe_tr_score, ppe_te_score))

        # Obtain predicted mean and stddev for PPE estimator
        # at each x value in provided dense grid of size G
        ppe_mean_G = ppe_estimator.predict(xgrid_G1)

        # Call predict_variance to get variance at each of the G test points
        # Then transform variance to stddev!!
        ppe_stddev_G = ppe_estimator.predict_variance(xgrid_G1)

        regr_viz_utils.plot_predicted_mean_with_filled_stddev_interval(
            cur_ppe_ax, # plot on PPE figure's current axes
            xgrid_G1, ppe_mean_G, ppe_stddev_G,
            num_stddev=2,
            color='b',
            legend_label='PPE +/- 2 stddev')

    regr_viz_utils.finalize_x_vs_t_plot(
        ppe_axgrid, x_train_ND[:N], t_train_N[:N], x_test_ND, t_test_N,
        order_list, alpha_list, beta_list)
    plt.show()
