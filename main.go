package main

import (
	"encoding/csv"
	"fmt"
	"os"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"github.com/petar/GoMNIST"
)

const numEpochs = 10

// Function to produce the CSV file
func produceCSV(trainFull training.Examples, neural *deep.Neural) {
	// Create or overwrite the CSV file
	csvFile, err := os.Create("./results/goDNNScores.csv")
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

	writer := csv.NewWriter(csvFile)
	defer writer.Flush()

	// Write CSV header
	writer.Write([]string{"Index", "Label", "LabelPred", "Accuracy"})

	for i, example := range trainFull {
		pred := neural.Predict(example.Input)
		predIdx := 0
		trueIdx := 0
		for j, val := range pred {
			if val > pred[predIdx] {
				predIdx = j
			}
		}
		for j, val := range example.Response {
			if val == 1.0 {
				trueIdx = j
				break
			}
		}
		accuracy := 0
		if predIdx == trueIdx {
			accuracy = 1
		}
		writer.Write([]string{fmt.Sprintf("%d", i), fmt.Sprintf("%d", trueIdx), fmt.Sprintf("%d", predIdx), fmt.Sprintf("%d", accuracy)})
	}

	fmt.Println("CSV file './results/goDNNScores.csv' has been created.")
}
func main() {
	// Load the MNIST dataset
	trainOriginalData, testOriginalData, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("First Train label: ", trainOriginalData.Labels[0])
	printImage(trainOriginalData.Images[0])

	// This code returns the train and test MNIST.Set types
	// Set has NRow, NCol, Images ([]RawImage), Labels ([]Label)

	fmt.Println("MNIST Rows: ", trainOriginalData.NRow, testOriginalData.NRow)
	fmt.Println("MNIST Columns: ", trainOriginalData.NCol, testOriginalData.NCol)
	xTrainData := convertMNISTForModeling(trainOriginalData.Images)
	yTrainData := convertLabelsForModeling(trainOriginalData.Labels)
	xTestData := convertMNISTForModeling(testOriginalData.Images)
	yTestData := convertLabelsForModeling(testOriginalData.Labels)
	printShape(xTrainData)
	printShape(yTrainData)
	printShape(xTestData)
	printShape(yTestData)

	trainingSet := training.Examples{}
	for i := range xTrainData {
		trainingSet = append(trainingSet, training.Example{Input: xTrainData[i], Response: yTrainData[i]})
	}

	train, valid := trainingSet.Split(0.7)

	trainFull := make(training.Examples, len(trainingSet))
	copy(trainFull, trainingSet)

	// Printing first 10 records of trainingSet
	// Debug printouts.  Helpful to see what the data looks like
	print1 := false
	if print1 {
		fmt.Println("First 10 records of trainingSet:")
		for i := 0; i < 10; i++ {
			fmt.Printf("Training Set Response: %v\n", trainingSet[i].Response)
			fmt.Printf("Training Full Response: %v\n", trainFull[i].Response)
			fmt.Println("Training Set Input")
			printInput(trainingSet[i].Input)
			fmt.Println("Training Full Input")
			printInput(trainFull[i].Input)
		}
	}

	testSet := training.Examples{}
	for i := range xTestData {
		testSet = append(testSet, training.Example{Input: xTestData[i], Response: yTestData[i]})
	}
	test := testSet

	neural := deep.NewNeural(&deep.Config{
		Inputs: len(train[0].Input),
		// Layout:     []int{32, 10},
		Layout:     []int{32, 64, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1), // slight positive bias helps ReLU
		Bias:       true,
	})

	//trainer := training.NewTrainer(training.NewSGD(0.01, 0.5, 1e-6, true), 1)
	//	trainer := training.NewBatchTrainer(training.NewAdam(0.02, 0.9, 0.999, 1e-8), 1, 200, 8)
	trainer := training.NewBatchTrainer(training.NewAdam(0.005, 0.9, 0.999, 1e-8), 1, 200, 8)

	fmt.Printf("training: %d, val: %d, test: %d\n", len(train), len(valid), len(test))

	// trainer.Train(neural, train, valid, 500)
	trainer.Train(neural, train, valid, numEpochs)

	// Calculate test accuracy
	testCorrect := 0
	for _, example := range test {
		pred := neural.Predict(example.Input)
		predIdx := 0
		trueIdx := 0
		for j, val := range pred {
			if val > pred[predIdx] {
				predIdx = j
			}
		}
		for j, val := range example.Response {
			if val == 1.0 {
				trueIdx = j
				break
			}
		}
		if predIdx == trueIdx {
			testCorrect++
		}
	}
	testAccuracy := float64(testCorrect) / float64(len(test)) * 100
	fmt.Printf("Test Accuracy: %.2f%%\n", testAccuracy)

	produceCSV(trainFull, neural)
}
