import matplotlib.pyplot as plt


# function to display original, model, and residual images for individual targets
def display_images(mod, chi, subimage):
    # make the residual image
    diff = subimage - mod

    # setup plotting
    fig = plt.figure(figsize=(7, 2))

    ax1 = fig.add_subplot(131, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))
    ax2 = fig.add_subplot(132, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))
    ax3 = fig.add_subplot(133, autoscale_on=False, xlim=(0, 17), ylim=(0, 17))

    ax1.set(xticks=[], yticks=[])
    ax2.set(xticks=[], yticks=[])
    ax3.set(xticks=[], yticks=[])

    # display the images
    im1 = ax1.imshow(subimage, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax1.set_title('Original Image')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(mod, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax2.set_title('Model')
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(diff, cmap='gray')  # , vmin = 0.01, vmax = 0.20
    ax3.set_title('Residual')
    fig.colorbar(im3, ax=ax3)

    return

# calling sequence display_images(tractor.getModelImage(0),tractor.getChiImage(0), subimage )
