# dataframe
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

print("First row: %s " % myarray[0])
print("last row: %s " % myarray[-1])


mylist = [1, 2, 3]
myarray2 = numpy.array(mylist)
print(myarray2)
print(myarray2.shape)

