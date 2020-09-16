import process
import models
import output

if __name__ == "__main__":
	result = process.train()
	output.Plot(result)
