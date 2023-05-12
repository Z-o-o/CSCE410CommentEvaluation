library(verification)
library(readr)
library(PRROC)
library(AUPRC)
library(ggplot2)
library(dplyr)

# Import data
test_output <- read_csv("./trained_true_pred.csv", show_col_types = FALSE)
untrained_test_output <- read_csv("./untrained_true_pred.csv", show_col_types = FALSE)

predicted <- test_output$pred
actual <- test_output$true

predicted_untrained <- untrained_test_output$pred
actual_untrained <- untrained_test_output$true

# PR Curve
## Trained & Untrained
png(file = "figures/PR_curve.png",
    width = 600,
    height = 600,
    res = 125)
pr <- pr.curve(scores.class0 = predicted, weights.class0 = +(!actual), curve = TRUE) # pos_label = 0
pr2 <- pr.curve(scores.class0 = predicted_untrained, weights.class0 = +(!actual_untrained), curve = TRUE)
trained_AUprC <- paste("Trained, AUC=", round(pr$auc.integral, 4))
untrained_AUprC <- paste("Untrained, AUC=", round(pr2$auc.integral, 4))
plot(pr, main = "Precision-Recall Curve", xlab = "Recall", ylab = "Precision", color = "#5C90D3")
plot(pr2, add = TRUE, color = "#f46524")
legend("topright", legend=c(trained_AUprC, untrained_AUprC),
       col=c("#5C90D3", "#f46524"), lty=1:1, cex=0.8)
dev.off()

# ROC Curve
## Trained & Untrained
png(file = "figures/ROC_curve.png",
    width = 600,
    height = 600,
    res = 125)
roc <- roc.curve(scores.class0 = predicted, weights.class0 = +(!actual), curve = TRUE) # pos_label = 0
roc2 <- roc.curve(scores.class0 = predicted_untrained, weights.class0 = +(!actual_untrained), curve = TRUE)
trained_AUroC <- paste("Trained, AUC=", round(roc$auc, 4))
untrained_AUroC <- paste("Untrained, AUC=", round(roc2$auc, 4))
plot(roc, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate", color = "#5C90D3")
plot(roc2, add=TRUE, color = "#f46524")
legend("topleft", legend=c(trained_AUroC, untrained_AUroC),
       col=c("#5C90D3", "#f46524"), lty=1:1, cex=0.8)
dev.off()


# Testing
fg <- rnorm(300)
bg <- rnorm(500, -2)

prtest <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
plot(prtest, main = "Precision-Recall Curve", xlab = "Recall", ylab = "Precision")


# Bar Plot for Results
bert_stats = c("Accuracy", "Sensitivity", "Specificity", "Precision", "NPV", "F1")
colors = c("#f46524", "#5C90D3")
models = c("Trained BERT", "Untrained BERT")

trained_bert_stats = c(65.79, 97.053, 3.637, 66.692, 38.3, 79.058)
untrained_bert_stats = c(41.14, 26.272, 70.601, 63.918, 32.572, 37.238)

png(file = "figures/stats_barplot.png",
    width = 1000,
    height = 500,
    res = 125)
barplot(rbind(trained_bert_stats, untrained_bert_stats),
        main = "Comparison of Metrics",
        names.arg = bert_stats,
        ylim = c(0, 100),
        xlab = "Stat",
        ylab = "Percentage",
        col = colors,
        beside = TRUE)
legend("topright", models,cex = .8, fill = colors)
dev.off()
