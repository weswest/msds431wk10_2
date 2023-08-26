package images

import (
	"fmt"
	"math"

	"github.com/petar/GoMNIST"
)

// This is related to GoMNIST
// Print the image to the console
func PrintImage(image GoMNIST.RawImage) {
	scaleFactor := 255.0 / 8.0
	nRow := 28
	nCol := 28

	for i := 0; i < nRow; i++ {
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
		// Start a new line after each row
		fmt.Println()
	}
}

func PrintInput(input []float64) {
	nRow := 28
	nCol := 28

	for i := 0; i < nRow; i++ {
		for j := 0; j < nCol; j++ {
			// Get the pixel value at the current position
			pixel := input[i*nCol+j]

			// Print a space if the pixel value is close to 0, otherwise print a "#"
			if pixel < 128 {
				fmt.Print(" ")
			} else {
				fmt.Print("#")
			}
		}
		// Start a new line after each row
		fmt.Println()
	}
}

// This takes all of the images and converts them to float64s
func ConvertMNISTForModeling(images []GoMNIST.RawImage) [][]float64 {
	var floatImages [][]float64

	for _, image := range images {
		var floatImage []float64
		for _, pixel := range image {
			floatImage = append(floatImage, float64(pixel))
		}
		floatImages = append(floatImages, floatImage)
	}

	return floatImages
}

func ConvertLabelsForModeling(labels []GoMNIST.Label) [][]float64 {
	var floatLabels [][]float64
	for _, label := range labels {
		oneHot := make([]float64, 10)
		oneHot[label] = 1.0
		floatLabels = append(floatLabels, oneHot)
	}
	return floatLabels
}

func PrintShape(matrix [][]float64) {
	rows := len(matrix)
	cols := 0
	if rows > 0 {
		cols = len(matrix[0])
	}
	fmt.Printf("Shape: %d x %d\n", rows, cols)
}
