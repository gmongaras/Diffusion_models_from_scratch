





# Reduce an image from the range (0, 255) -> (-1, 1)
def reduce_image(X):
    return ((X-127.5)/127.5)


# Unreduce an image from the range (-1, 1) -> (0, 255)
def unreduce_image(X):
    return (X*127.5)+127.5

