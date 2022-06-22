package cipherUtils

import (
	"github.com/ldsec/dnn-inference/inference/plainUtils"
	"github.com/tuneinsight/lattigo/v3/ckks"
	"math"
)

const FILLTHRESHOLD = 0.5 //how many % of slots out of available slots should be filled -> heuristic base

type BlockSplits struct {
	InnerRows, InnerCols, RowP, ColP int
}

func GetFillRatio(rows, cols, replicaFactor int, slotsAvailable float64) float64 {
	consumedSlots := float64(rows * cols * replicaFactor)
	return consumedSlots / slotsAvailable
}

//Computes the optimal number of rows for input sub-matrices. Takes the innerCols of the Input and the maxInnerCols of all the weights in the pipeline
func GetOptimalInnerRows(inputInnerCols int, maxInnerCols int, params ckks.Parameters) int {
	innerCols := plainUtils.Max(inputInnerCols, maxInnerCols)
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots()-1)))
	optInnerRows := int(math.Floor(slotsAvailable / float64(innerCols)))
	//takes into account that if maxInnerCols > inputInnerRows we will have to rotate during prepacking with 3x space occupied
	if optInnerRows*inputInnerCols*3 > params.LogSlots() {
		return optInnerRows
	} else {
		for optInnerRows*inputInnerCols*3 > params.LogSlots() {
			optInnerRows--
		}
		return optInnerRows
	}
}

// Finds all possible splits for the current model, given the number
// of input features (e.g 784 in case of MNIST) and the dimentions of all
// the weights of the model expressed as matrices
func FindSplits(inputFeatures int, weightRows, weightCols []int, params ckks.Parameters, strategyOnBatch bool) [][]BlockSplits {
	sq := int(math.Ceil(math.Sqrt(float64(inputFeatures))))
	var colPartitions []int
	var innerCols []int
	var batchSizes []int
	slotsAvailable := float64(math.Pow(2, float64(params.LogSlots())))

	for d := 2; d <= sq; d++ {
		if inputFeatures%d == 0 {
			colPartitions = append(colPartitions, d)
			innerCols = append(innerCols, inputFeatures/d)
			batch := int(math.Floor(slotsAvailable / (2 * float64(inputFeatures/d))))
			if batch*(inputFeatures/d)*2 <= int(slotsAvailable) {
				batchSizes = append(batchSizes, batch)
			}
		}
	}

	//simulate to see if the splits work with the params

	var blockSplits [][]BlockSplits

	for i := range batchSizes {
		batch := batchSizes[i]
		colP := colPartitions[i]
		inCols := innerCols[i]
		currCols := inCols
		currColP := colP
		isValid := true

		blockSplit := make([]BlockSplits, len(weightRows)+1) //input + weights

		for w := range weightRows {
			inRowsW := weightRows[w] / currColP
			inColsW := plainUtils.Min(int(math.Ceil(slotsAvailable/float64(inRowsW))), weightCols[w])
			for {
				//adjust weight inner col split or batch depending on strategy
				adjusted := true
				//check if weight submatrix can be stored and that the fillratio > threshold (also taking into account the max possible size of this weight)
				if inRowsW*inColsW > int(slotsAvailable) || GetFillRatio(inRowsW, inColsW, 1, float64(plainUtils.Min(int(slotsAvailable), weightRows[w]*weightCols[w]))) < FILLTHRESHOLD {
					adjusted = false
				}
				//check if replication of input can be stored

				if GetReplicaFactor(inRowsW, inColsW)*batch*currCols > int(slotsAvailable) || GetFillRatio(batch, currCols, GetReplicaFactor(inRowsW, inColsW), slotsAvailable) < FILLTHRESHOLD {
					adjusted = false
				}
				if !adjusted {
					if strategyOnBatch {
						// find nearest divisors of weight column
						inColsW--
						//loop until divisor of lower then 1
						for weightCols[w]%inColsW != 0 && inColsW > 1 {
							inColsW--
						}
					} else {
						//decrease batch
						batch--
					}
				}
				if adjusted {
					break
				}
				if strategyOnBatch {
					if inColsW <= 1 {
						isValid = false
						break
					}
				} else {
					if batch < 1 {
						isValid = false
						break
					}
				}
			}
			blockSplit[w+1] = BlockSplits{InnerRows: inRowsW, InnerCols: inColsW, RowP: currColP, ColP: weightCols[w] / inColsW}
			currColP = weightCols[w] / inColsW
			currCols = inColsW
		}
		if isValid {
			blockSplit[0] = BlockSplits{InnerRows: batch, InnerCols: inCols, RowP: 1, ColP: colP}
			blockSplits = append(blockSplits, blockSplit)
		}
	}
	return blockSplits
}
