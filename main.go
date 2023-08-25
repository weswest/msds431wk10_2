package main

import (
	"fmt"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"github.com/petar/GoMNIST"
)

func main() {
	// Load the MNIST dataset
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("First Train label: ", train.Labels[0])
	printImage(train.Images[0])

	// This code returns the train and test MNIST.Set types
	// Set has NRow, NCol, Images ([]RawImage), Labels ([]Label)

	fmt.Println("MNIST Rows: ", train.NRow, test.NRow)
	fmt.Println("MNIST Columns: ", train.NCol, test.NCol)
	inputData := convertMNISTForModeling(train.Images)
	outputData := convertLabelsForModeling(train.Labels)
	printShape(inputData)
	printShape(outputData)

	// Define the neural network
	n := deep.NewNeural(&deep.Config{
		Inputs:     784,                 // 28x28
		Layout:     []int{512, 256, 10}, // 3 layers: 2 hidden layers and 1 output layer
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewUniform(0.5, 0.0),
		Bias:       true,
	})

	// Train the neural network
	optimizer := training.NewSGD(0.01, 0.1, 1e-6, true)
	trainer := training.NewTrainer(optimizer, 50)
	trainingSet := training.Examples{}
	for i := range inputData {
		trainingSet = append(trainingSet, training.Example{Input: inputData[i], Response: outputData[i]})
	}
	fmt.Println("Training set length: ", len(trainingSet))
	training, validation := trainingSet.Split(0.02)
	fmt.Println("Just about to train")
	fmt.Println("Training set length after train/valid split: ", len(training))
	trainer.Train(n, training, validation, 3) // 10 epochs for demonstration
	fmt.Println("After trainer.Train")

	// Evaluate accuracy on test data
	testInputData := convertMNISTForModeling(test.Images)
	testOutputData := convertLabelsForModeling(test.Labels)
	correct := 0
	for i, input := range testInputData {
		pred := n.Predict(input)
		predIdx := 0
		trueIdx := 0
		for j, val := range pred {
			if val > pred[predIdx] {
				predIdx = j
			}
		}
		for j, val := range testOutputData[i] {
			if val == 1.0 {
				trueIdx = j
				break
			}
		}
		if predIdx == trueIdx {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testInputData))
	fmt.Printf("Accuracy: %f\n", accuracy)
}
