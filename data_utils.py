import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data import BDDObjectDetectionDataset


def get_dataloader_idx_from_df(dataset_obj: BDDObjectDetectionDataset, df_id: int) -> int:
    """
    Get the dataloader index from the dataframe index.
    """
    return np.where(dataset_obj.image_ids == dataset_obj.data_df.iloc[df_id].image_id)[0].tolist()[0]

def plot_image_with_data(image, target, class_names):
    """
    Plot the image with bounding boxes and labels.
    Class names are used to display the labels.
    """
    if target['boxes']:
        plt.imshow(image)
        ax = plt.gca()
        for box_id, box in enumerate(target['boxes']):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='g', facecolor='none', alpha=0.4)
            # Add label to the box
            ax.text(box[0], box[1], class_names[target['labels'][box_id]], color='r')
            ax.add_patch(rect)
        plt.show()
    else:
        print('No objects in the image')

def plot_random_images(dataset_obj: BDDObjectDetectionDataset, num_images=3, cat_name=None, img_attr_name=None, ann_attr_name=None):
    """
    Plot random images from the dataset.
    You can filter the images by category name, image attribute name, and annotation attribute name.
    img_attr_name and ann_attr_name should be in the format 'attribute_name.attribute_value'.
    """
    curr_df = dataset_obj.data_df
    print(f'DF shape: {curr_df.shape}')
    if cat_name:
        curr_df = dataset_obj.data_df[dataset_obj.data_df.category_name == cat_name]
        print(f'DF shape after category filtering: {curr_df.shape}')
    if img_attr_name:
        curr_df = curr_df[curr_df.image_attributes.apply(lambda x: str(x.get(img_attr_name.split('.')[0])) == img_attr_name.split('.')[1])]
        print(f'DF shape after image attribute filtering: {curr_df.shape}')
    if ann_attr_name:
        curr_df = curr_df[curr_df.ann_attributes.apply(lambda x: str(x.get(ann_attr_name.split('.')[0])) == ann_attr_name.split('.')[1])]
        print(f'DF shape after annotation attribute filtering: {curr_df.shape}')
    if len(curr_df) == 0:
        print('No images found with the specified filters')
        return
    # Get num_images number of random ids from the curr_df index
    rand_df_idx = np.random.choice(curr_df.index, num_images)
    for i in rand_df_idx:
        idx = get_dataloader_idx_from_df(dataset_obj, i)
        plot_image_with_data(*dataset_obj[idx], dataset_obj.class_names)

def plot_distributions(series_data: pd.Series, title='', xlabel='', ylabel=''):
    """
    Plot the distribution of a pandas Series. Calculate the counts and percentages of each unique value.
    title, xlabel, and ylabel are used to customize the plot.
    """
    counts = series_data.value_counts().sort_values(ascending=False)
    
    # Calculate percentages
    total = len(series_data)
    percentages = counts / total * 100

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values)

    # Add labels on top of each bar
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        formatted_count = f"{count:,}"  # Add commas to the count
        ax.text(i, count, f'{formatted_count}\n({percentage:.1f}%)', 
                ha='center', va='bottom')

    # Customize the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_attribute_distribution(df: pd.DataFrame, attribute_name: str, column_name: str = 'image_attributes', split=''):
    """
    Plots the distribution of a specific attribute from the 'column_name' column.
    Column name can be 'image_attributes' or 'ann_attributes'.
    Split is used to customize the title.
    """

    try:
        attribute_series = df[column_name].apply(lambda x: x.get(attribute_name))
        plot_distributions(attribute_series, 
                         title=f'{split} Distribution of {attribute_name}',
                         xlabel=attribute_name,
                         ylabel='Count')
    except AttributeError:
        print(f"Error: {column_name} column likely contains non-dictionary values. Check your data.")
    except KeyError:
        print(f"Error: Attribute '{attribute_name}' not found in {column_name} dictionaries.")

def plot_bbox_scatter(df, title="BBox Width vs. Height"):
    """
    Plot a scatter plot of BBox width vs. height.
    Also shows the regression line and a linear line (y=x) for comparison.
    """
    df = df.copy()  # Create a copy of the original df to avoid warnings
    df.loc[:, 'bbox_width'] = df['bbox'].apply(lambda x: x[2] - x[0])  # X2-X1
    df.loc[:, 'bbox_height'] = df['bbox'].apply(lambda x: x[3] - x[1])  # Y2-Y1

    plt.figure(figsize=(8, 6))
    plt.scatter(df['bbox_width'], df['bbox_height'], alpha=0.5, s=10)  # alpha for transparency
    plt.xlabel("BBox Width")
    plt.ylabel("BBox Height")
    plt.title(title)
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()
    z = np.polyfit(df['bbox_width'], df['bbox_height'], 1)  # 1 for linear fit
    p = np.poly1d(z)
    # Plot the regression line
    plt.plot(df['bbox_width'], p(df['bbox_width']), "r--", label="Regression Line") #r-- is red and dashed
    x_values = np.linspace(df['bbox_width'].min(), df['bbox_width'].max(), 100)
    plt.plot(x_values, x_values, color='purple', linestyle=':', label=f"Linear Line (y=x)")

    plt.legend() # Show the legend
    plt.show()
