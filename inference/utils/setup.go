package utils

import (
	"errors"
	"fmt"
	"log"
	"os"
)

func SetupDirectory() {
	if _, err := os.Stat(os.ExpandEnv("$HOME/gef/keys")); errors.Is(err, os.ErrNotExist) {
		if _, err := os.Stat(os.ExpandEnv("$HOME/gef")); errors.Is(err, os.ErrNotExist) {
			err := os.Mkdir(os.ExpandEnv("$HOME/gef"), os.ModePerm)
			if err != nil {
				log.Println(err)
			}
		}
		fmt.Println("Creating keys directory at " + os.ExpandEnv("$HOME/gef/keys"))
		err := os.Mkdir(os.ExpandEnv("$HOME/gef/keys"), os.ModePerm)
		if err != nil {
			log.Println(err)
		}
	}
	fmt.Println("Key Directory : OK")
}
