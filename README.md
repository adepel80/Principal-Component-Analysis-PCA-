# Principal-Component-Analysis-PCA

## Project review
This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction, noise reduction, and data visualization. PCA is a powerful technique in machine learning that transforms high-dimensional data into a lower-dimensional form while preserving as much variability as possible. This project includes practical examples of PCA applied to various datasets, highlighting its benefits and applications.

## Libraries
![ENSE LIBRAROES](https://github.com/adepel80/Principal-Component-Analysis-PCA-/assets/123180341/bab0218f-05ff-43f9-b5bc-ba289e149312)


# Data
The project utilizes several datasets to demonstrate the versatility of PCA. The datasets are stored in the data/ directory, with raw data in the data/raw subdirectory and processed data in the data/processed subdirectory.

```
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2), rng.randn(2,200)).T
```

## Results
The project demonstrates how PCA can effectively reduce the dimensionality of datasets, highlight underlying structures, and improve data visualization. Detailed performance metrics, variance explained by principal components, and visualizations are included in the reports/pca_report.pdf file.

#### PCA - VARIANCE
![pca visual1](https://github.com/adepel80/Principal-Component-Analysis-PCA-/assets/123180341/57fa297e-4871-4e34-9622-544507ccc6b9)


### Reduction from 64 Dimension to 2
![PCA 64 -2](https://github.com/adepel80/Principal-Component-Analysis-PCA-/assets/123180341/f56a9511-2b3b-47a7-9d7e-09d26e4cef61)

``` pca = PCA(2) #Project from 64 to 2 dimensions
pca.fit(digits.data)
projected = pca.transform(digits.data)
print(digits.data.shape)
print(projected.shape)
```

### Choosing the number of component
![pca choosing nos of comp](https://github.com/adepel80/Principal-Component-Analysis-PCA-/assets/123180341/913ad39f-6d89-46a2-b91e-3b51d4764a16)

``` 
pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
```



## Key Findings

### Dimensionality Reduction:
PCA significantly reduces the number of features while retaining the essential information.
### Data Visualization: 
Lower-dimensional representations make it easier to visualize and interpret complex datasets.
### Noise Reduction:
PCA effectively reduces noise in the data, leading to better performance in subsequent machine learning models.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.

This description provides a clear and structured overview of the PCA project, highlighting the techniques used, the datasets, key findings, and practical information for installation and usage.
