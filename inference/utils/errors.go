package utils

import (
	"log"
)

func ThrowErr(err error) {
	if err != nil {
		log.Println(err)
		panic(err)
	}
}
