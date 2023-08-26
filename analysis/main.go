package main

import (
	"encoding/csv"
	"fmt"
	"os"
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

func main() {
	dnnRecords, anomalyRecords := readInScores()
	compareRecords(dnnRecords, anomalyRecords)
}
