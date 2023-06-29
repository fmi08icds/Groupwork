import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


file =  "/Users/abdulnaser/Desktop/Groupwork/Group2/data/diabetes_prediction_dataset.csv"
df = pd.read_csv(file)

def plot_2d_visualization(df,x, y,to_save_name):
    # Set the plot size
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    sns.scatterplot(data=df, x=x, y=y, hue='diabetes')

    # Set labels and title
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} with Diabetes')

    # Save the plot
    plt.savefig(f"/Users/abdulnaser/Desktop/Groupwork/Group2/src/visualization/" + to_save_name )

    # Show the plot
    plt.show()


def plot_3d_visualization(df,x,y,z,output_file):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assign different colors based on the 'diabetes' variable
    colors = {0: 'blue', 1: 'red'}
    df['color'] = df['diabetes'].map(colors)

    # Plot the 3D scatter plot
    ax.scatter(df[x], df[y], df[z], c=df['color'], marker='o')

    # Set labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('3D Scatter Plot with Diabetes')

    # Save the plot
    plt.savefig(output_file)

    # Show the plot
    plt.show()




plot_2d_visualization(df, 'age','bmi','age_vs_bmi.png')
plot_2d_visualization(df, 'blood_glucose_level','bmi','blood_glucose_level_vs_bmi.png')
plot_3d_visualization(df, 'age', 'bmi', 'blood_glucose_level','age_vs_bmi_vs_blood_glucose_level.png')