
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib

# 加载模型和验证数据集
X_test, y_test = joblib.load('../processing data/test.joblib')
# probabilities from the logistic regression model
lr_test_probs = joblib.load('../test_models/lr_test_probs.joblib')
# probabilities from the random forest model
rf_test_probs = joblib.load('../test_models/rf_test_probs.joblib')
# probabilities from the xgboost model
xgb_test_probs = joblib.load('../test_models/xgb_test_probs.joblib')


# 计算每个模型的ROC曲线
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_test_probs)

# 计算AUC值
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)


# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc_lr), color='green')
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(auc_rf), color='blue')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(auc_xgb), color='orange')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--')

# 添加轴标签和图例
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Four Models in Test Data')
plt.legend(loc="lower right")

# 显示图形
plt.show()

