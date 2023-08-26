package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/petar/GoMNIST"
)

// readCSVFiles reads the DNN and anomaly CSV files and returns their records.
func readInScores() ([][]string, [][]string) {
	// Open the DNN CSV file
	dnnFile, err := os.Open("./results/goDNNScores.csv")
	if err != nil {
		panic(err)
	}
	defer dnnFile.Close()

	// Open the anomaly CSV file
	anomalyFile, err := os.Open("./results/goIForestScores.csv")
	if err != nil {
		panic(err)
	}
	defer anomalyFile.Close()

	// Read the DNN CSV file
	dnnReader := csv.NewReader(dnnFile)
	dnnRecords, err := dnnReader.ReadAll()
	if err != nil {
		panic(err)
	}

	// Read the anomaly CSV file
	anomalyReader := csv.NewReader(anomalyFile)
	anomalyRecords, err := anomalyReader.ReadAll()
	if err != nil {
		panic(err)
	}

	return dnnRecords, anomalyRecords
}

// compareRecords compares the labels in the DNN and anomaly records and prints the results.
func compareRecords(dnnRecords [][]string, anomalyRecords [][]string) {
	// Check if the number of records in both files are the same
	if len(dnnRecords) != len(anomalyRecords) {
		fmt.Println("The number of records in the two files are not the same.")
		return
	}

	// Iterate over the records and compare the labels
	matchCount := 0
	noMatchCount := 0
	for i := 1; i < len(dnnRecords); i++ { // Start from 1 to skip the header row
		dnnLabel := dnnRecords[i][1]
		anomalyLabel := anomalyRecords[i][1]

		if dnnLabel == anomalyLabel {
			matchCount++
		} else {
			noMatchCount++
		}
	}

	// Report the results
	fmt.Printf("Number of indexes that Match: %d  Don't Match: %d\n", matchCount, noMatchCount)
}
func calculateAnomalyScores(dnnRecords [][]string, anomalyRecords [][]string, anomalyScoreType string) {
	// Verify the anomaly score type
	var anomalyScoreColumn int
	if anomalyScoreType == "iForestAnomalyScore" {
		anomalyScoreColumn = 2
		fmt.Println("Anomaly Score Type: iForestAnomalyScore")
	} else if anomalyScoreType == "rForestNormalizedScore" {
		anomalyScoreColumn = 3
		fmt.Println("Anomaly Score Type: rForestNormalizedScore")
	} else {
		fmt.Println("The selected anomaly score type is not valid. Please select either 'iForestAnomalyScore' or 'rForestNormalizedScore'.")
		return
	}

	// Initialize data structures to store the sum and count of anomaly scores for each label
	type LabelData struct {
		CorrectSum     float64
		IncorrectSum   float64
		CorrectCount   int
		IncorrectCount int
	}
	labelScores := make(map[int]*LabelData)

	// Iterate over the records and accumulate the anomaly scores
	for i := 1; i < len(dnnRecords); i++ { // Start from 1 to skip the header row
		label, _ := strconv.Atoi(dnnRecords[i][1])
		accuracy, _ := strconv.Atoi(dnnRecords[i][3])
		anomalyScore, _ := strconv.ParseFloat(anomalyRecords[i][anomalyScoreColumn], 64)

		if _, exists := labelScores[label]; !exists {
			labelScores[label] = &LabelData{}
		}

		if accuracy == 1 {
			labelScores[label].CorrectSum += anomalyScore
			labelScores[label].CorrectCount++
		} else {
			labelScores[label].IncorrectSum += anomalyScore
			labelScores[label].IncorrectCount++
		}
	}

	// Print the table
	fmt.Println("Label | Avg Anomaly Score (Correct) | Avg Anomaly Score (Incorrect)")
	fmt.Println("-------------------------------------------------------------")
	for label := 0; label <= 9; label++ {
		data := labelScores[label]
		avgCorrect := data.CorrectSum / float64(data.CorrectCount)
		avgIncorrect := data.IncorrectSum / float64(data.IncorrectCount)
		fmt.Printf("%d     | %.3f                        | %.3f\n", label, avgCorrect, avgIncorrect)
	}
}

func findImages(label int, anomalyScoreType string, trainData *GoMNIST.Set, dnnRecords [][]string, anomalyRecords [][]string) (int, int, int, int) {
	var correctHigh, correctLow, incorrectHigh, incorrectLow int
	var thresholdHigh, thresholdLow float64

	anomalyScoreIdx := 2 // Default to iForestAnomalyScore
	if anomalyScoreType == "rForestNormalizedScore" {
		anomalyScoreIdx = 3
	}

	// Convert records to numeric values and calculate thresholds
	allScores := make([]float64, len(anomalyRecords)-1) // -1 to exclude header
	for i := 1; i < len(anomalyRecords); i++ {
		score, _ := strconv.ParseFloat(anomalyRecords[i][anomalyScoreIdx], 64)
		allScores[i-1] = score
	}
	sort.Float64s(allScores)
	thresholdHigh = allScores[int(0.8*float64(len(allScores)))]
	thresholdLow = allScores[int(0.2*float64(len(allScores)))]

	// Iterate over the records to find the images
	for i, record := range dnnRecords[1:] { // Skip header
		actualLabel, _ := strconv.Atoi(record[1])
		isCorrect, _ := strconv.Atoi(record[3])
		anomalyScore, _ := strconv.ParseFloat(anomalyRecords[i+1][anomalyScoreIdx], 64)

		if actualLabel != label {
			continue
		}

		if isCorrect == 1 && correctHigh == 0 && anomalyScore >= thresholdHigh {
			correctHigh = i
		} else if isCorrect == 1 && correctLow == 0 && anomalyScore <= thresholdLow {
			correctLow = i
		} else if isCorrect == 0 && incorrectHigh == 0 && anomalyScore >= thresholdHigh {
			incorrectHigh = i
		} else if isCorrect == 0 && incorrectLow == 0 && anomalyScore <= thresholdLow {
			incorrectLow = i
		}

		// Break if all four images are found
		if correctHigh != 0 && correctLow != 0 && incorrectHigh != 0 && incorrectLow != 0 {
			break
		}
	}

	return correctHigh, correctLow, incorrectHigh, incorrectLow
}

func printImagesForLabelAndAnomalyScore(label int, anomalyScoreType string, trainData *GoMNIST.Set, dnnRecords [][]string, anomalyRecords [][]string) {
	// Verify the anomaly score type
	var anomalyScoreColumn int
	if anomalyScoreType == "iForestAnomalyScore" {
		anomalyScoreColumn = 2
	} else if anomalyScoreType == "rForestNormalizedScore" {
		anomalyScoreColumn = 3
	} else {
		fmt.Println("The selected anomaly score type is not valid. Please select either 'iForestAnomalyScore' or 'rForestNormalizedScore'.")
		return
	}

	// Filter and sort the dataset based on the anomaly scores
	type ImageData struct {
		Index        int
		Image        GoMNIST.RawImage
		AnomalyScore float64
		IsCorrect    bool
	}
	var filteredData []ImageData

	for i, record := range dnnRecords[1:] { // Skip header
		if int(trainData.Labels[i]) == label {
			anomalyScore, _ := strconv.ParseFloat(anomalyRecords[i+1][anomalyScoreColumn], 64)
			isCorrect := record[3] == "1"
			filteredData = append(filteredData, ImageData{
				Index:        i,
				Image:        trainData.Images[i],
				AnomalyScore: anomalyScore,
				IsCorrect:    isCorrect,
			})
		}
	}

	sort.Slice(filteredData, func(i, j int) bool {
		return filteredData[i].AnomalyScore < filteredData[j].AnomalyScore
	})

	// Identify the images
	top20Index := int(0.8 * float64(len(filteredData)))
	bottom20Index := int(0.2 * float64(len(filteredData)))

	var topCorrect, topIncorrect, bottomCorrect, bottomIncorrect GoMNIST.RawImage

	for _, data := range filteredData[:bottom20Index] {
		if data.IsCorrect && bottomCorrect == nil {
			bottomCorrect = data.Image
		} else if !data.IsCorrect && bottomIncorrect == nil {
			bottomIncorrect = data.Image
		}
	}

	for _, data := range filteredData[top20Index:] {
		if data.IsCorrect && topCorrect == nil {
			topCorrect = data.Image
		} else if !data.IsCorrect && topIncorrect == nil {
			topIncorrect = data.Image
		}
	}

	// Find the images based on the criteria
	correctHigh, correctLow, incorrectHigh, incorrectLow := findImages(label, anomalyScoreType, trainData, dnnRecords, anomalyRecords)

	// Print header
	fmt.Printf("Showing label %d images scored using the %s\n", label, anomalyScoreType)
	fmt.Printf("Image IDs: upper left (UL): %d, BL: %d, UR: %d, and BR: %d\n", correctHigh, correctLow, incorrectHigh, incorrectLow)

	// Print column headers for the first row
	fmt.Println("Correct Prediction; High Anomaly Score\t\tIncorrect Prediction (Predicted:", dnnRecords[incorrectHigh][2], "); High Anomaly Score")
	PrintImageSideBySide(trainData.Images[correctHigh], trainData.Images[incorrectHigh])

	// Print column headers for the second row
	fmt.Println("Correct Prediction; Low Anomaly Score\t\tIncorrect Prediction (Predicted:", dnnRecords[incorrectLow][2], "); Low Anomaly Score")
	PrintImageSideBySide(trainData.Images[correctLow], trainData.Images[incorrectLow])
}

func PrintImageSideBySide(image1, image2 GoMNIST.RawImage) {
	scaleFactor := 255.0 / 8.0
	nRow := 28
	nCol := 28

	for i := 0; i < nRow; i++ {
		for _, image := range []GoMNIST.RawImage{image1, image2} {
			for j := 0; j < nCol; j++ {
				// Get the pixel value at the current position
				pixel := image[i*nCol+j]

				// Scale the pixel value
				scaledPixel := int(math.Round(float64(pixel) / scaleFactor))

				// Make sure that only 0 scales to 0
				if pixel != 0 && scaledPixel == 0 {
					scaledPixel = 1
				}

				// Print a space if the pixel value is 0, otherwise print the scaled pixel value
				if scaledPixel == 0 {
					fmt.Print(" ")
				} else {
					fmt.Print(scaledPixel)
				}
			}
			fmt.Print("\t\t") // Space between the two images
		}
		// Start a new line after each row
		fmt.Println()
	}
}

func main() {
	// Define and parse the label flag
	labelPtr := flag.Int("label", 3, "an integer representing the label")
	flag.Parse()

	// Validate the label value
	if *labelPtr < 0 || *labelPtr > 9 {
		fmt.Println("Error: The value for -label needs to be between 0-9.")
		return
	}
	// Load the MNIST dataset
	trainData, _, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(err)
	}

	dnnRecords, anomalyRecords := readInScores()
	compareRecords(dnnRecords, anomalyRecords)
	calculateAnomalyScores(dnnRecords, anomalyRecords, "iForestAnomalyScore")
	calculateAnomalyScores(dnnRecords, anomalyRecords, "rForestNormalizedScore")
	printImagesForLabelAndAnomalyScore(*labelPtr, "iForestAnomalyScore", trainData, dnnRecords, anomalyRecords)
	printImagesForLabelAndAnomalyScore(*labelPtr, "rForestNormalizedScore", trainData, dnnRecords, anomalyRecords)
}
