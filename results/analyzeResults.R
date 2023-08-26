# compare the anomaly scores from Python and R
digitLabels = read.csv("results/labels.csv")
scoresSolitudeR = read.csv("results/solitudeRScores.csv")
scoresIsotreeR = read.csv("results/isotreeRScores.csv")
scoresPython = read.csv("results/pythonScores.csv")
scoresGo = read.csv("results/goIForestScores.csv")
  
# merge the scoring data  
analyzeData <- data.frame("digitLabel" = digitLabels$digitLabel,
	"scoreSolitudeR" = scoresSolitudeR$iforestRScore.anomaly_score,
	"scoreIsotreeR" = scoresIsotreeR$isotreeRScore,
	"scorePython" = scoresPython$iforestPythonScore,
	"scoreGoIForest" = scoresGo$iForestAnomalyScore,
	"scoreGoRForest" = scoresGo$rForestNormalizedScore)

# Note that distributions of anomaly scores have different shapes
# Are there hyperparameter settings that may bring the 
# Python and R results closer together?
png(filename = "results/fig-python-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(density(scorePython)))
dev.off()

png(filename = "results/fig-r-solitude-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(density(scoreSolitudeR)))
dev.off()

png(filename = "results/fig-r-isotree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(density(scoreIsotreeR)))
dev.off()

png(filename = "results/fig-scatterplot-solitude-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scorePython,scoreSolitudeR))
title(paste("Correlation between Python and R solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scorePython,scoreSolitudeR)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-isotree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scorePython,scoreIsotreeR))
title(paste("Correlation between Python and R isotree anomaly scores:",
	as.character(round(with(analyzeData,cor(scorePython,scoreIsotreeR)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-isotree-solitude-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreIsotreeR, scoreSolitudeR))
title(paste("Correlation between R isotree and Solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreIsotreeR, scoreSolitudeR)),digits = 2))))
dev.off()

png(filename = "results/fig-go-itree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(density(scoreGoIForest)))
dev.off()

png(filename = "results/fig-scatterplot-goIF-x-python-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoIForest,scorePython))
title(paste("Correlation between Go IF and Python anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoIForest, scorePython)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-goIF-x-risotree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoIForest,scoreIsotreeR))
title(paste("Correlation between Go IForest and rIsotree anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoIForest, scoreIsotreeR)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-goIF-x-solitude-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoIForest,scoreSolitudeR))
title(paste("Correlation between Go IForest and r Solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoIForest, scoreSolitudeR)),digits = 2))))
dev.off()

png(filename = "results/fig-go-rtree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(density(scoreGoRForest)))
dev.off()

png(filename = "results/fig-scatterplot-goRF-x-python-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoRForest,scorePython))
title(paste("Correlation between Go RForest and Python anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoRForest, scorePython)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-goRF-x-risotree-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoRForest,scoreIsotreeR))
title(paste("Correlation between Go RForest and rIsotree anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoRForest, scoreIsotreeR)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-goRF-x-solitude-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoRForest,scoreSolitudeR))
title(paste("Correlation between Go RForest and r Solitude anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoRForest, scoreSolitudeR)),digits = 2))))
dev.off()

png(filename = "results/fig-scatterplot-goRF-x-goIFe-anomaly-scores.png", width = 11, height = 8.5, units = "in", res = 300)
with(analyzeData, plot(scoreGoRForest,scoreGoIForest))
title(paste("Correlation between Go RForest and Go IForest anomaly scores:",
	as.character(round(with(analyzeData,cor(scoreGoRForest, scoreGoIForest)),digits = 2))))
dev.off()
