# import numpy as np
# import logging
# from abc import ABC, abstractmethod
# import pandas as pd
# from typing import Union
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.decomposition import PCA
# from sentence_transformers import SentenceTransformer
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # Download NLTK data (only needs to be run once)
# nltk.download('stopwords')
# nltk.download('wordnet')

# class DataStrategy(ABC):
#     '''
#         Strategy for data handling
#     '''
#     @abstractmethod
#     def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
#         pass

# class DataPreProcessing(DataStrategy):
#     def __init__(self):
#         self.model = SentenceTransformer('bert-base-nli-mean-tokens')
#         self.stop_words = set(stopwords.words('english'))
#         self.lemmatizer = WordNetLemmatizer()
#         self.scaler = StandardScaler()
#         self.encoder = OneHotEncoder( handle_unknown='ignore')

#     def clean_text(self, text: str) -> str:
#         text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#         words = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words]
#         return " ".join(words)

#     def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         logging.info("Processing Data")
#         try:
#             data = data.drop_duplicates()

#             # Drop unnecessary date-related columns
#             date_columns = [
#                 "order_approved_at", "order_delivered_carrier_date",
#                 "order_delivered_customer_date", "order_estimated_delivery_date",
#                 "order_purchase_timestamp"
#             ]
#             data = data.drop(date_columns, axis=1, errors='ignore')

#             # Drop ID columns that are not useful for modeling
#             id_columns = ["customer_id", "order_id", "product_id", "seller_id","customer_unique_id","order_item_id","review_id",]
#             data = data.drop(id_columns, axis=1, errors='ignore')

#             # Handle missing values for numerical columns
#             for column in data.columns:
#                 if data[column].dtype in ['float64', 'int64']:
#                     median_value = data[column].median()
#                     data[column].fillna(median_value, inplace=True)

#             # Handle missing values for object columns
#             for column in data.columns:
#                 if data[column].dtype == 'object':
#                     data[column].fillna("Unknown", inplace=True)

#             # Clean and encode the review comments
#             data["review_comment_message"] = data["review_comment_message"].fillna("No review")
#             data["cleaned_review_comment"] = data["review_comment_message"].apply(self.clean_text)

#             bert_embeddings = self.model.encode(data["cleaned_review_comment"].tolist())
#             bert_embeddings_df = pd.DataFrame(
#                 bert_embeddings,
#                 columns=[f'embedding_{i}' for i in range(bert_embeddings.shape[1])]
#             )

#             data = pd.concat([data.reset_index(drop=True), bert_embeddings_df], axis=1)
#             data = data.drop(columns=["review_comment_message", "cleaned_review_comment"])

#             # Separate the target variable
#             if 'review_score' in data.columns:
#                 review_score = data['review_score'].reset_index(drop=True)
#                 data = data.drop('review_score', axis=1)
#             else:
#                 logging.error("'review_score' column not found in data.")
#                 raise KeyError("'review_score' column not found in data.")

#             # Identify non-numeric columns
#             non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()

#             # Encode categorical variables if any
#             if non_numeric_cols:
#                 encoded_categorical = self.encoder.fit_transform(data[non_numeric_cols])
#                 encoded_categorical_df = pd.DataFrame(
#                     encoded_categorical,
#                     columns=self.encoder.get_feature_names_out(non_numeric_cols)
#                 )
#                 data = pd.concat([data.reset_index(drop=True), encoded_categorical_df], axis=1)
#                 data = data.drop(non_numeric_cols, axis=1)

#             # Drop any remaining non-numeric columns (if any)
#             data = data.select_dtypes(include=[np.number])

#             # Standardize numeric features
#             numeric_cols = data.columns
#             data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])

#             # PCA to handle collinearity
#             pca = PCA(n_components=0.95, random_state=42)
#             pca_data = pca.fit_transform(data)
#             pca_data_df = pd.DataFrame(
#                 pca_data,
#                 columns=[f'PCA_{i}' for i in range(pca_data.shape[1])]
#             )

#             # Add the target variable back to the DataFrame
#             final_data = pd.concat([pca_data_df.reset_index(drop=True), review_score], axis=1)

#             return final_data

#         except Exception as e:
#             logging.error(f'Exception Occurred - {e}')
#             raise e

# class DataDivideToTrainAndTest(DataStrategy):
#     def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
#         try:
#             X = data.drop(["review_score"], axis=1)
#             y = data["review_score"]  # target variable column
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )
#             return X_train, X_test, y_train, y_test
#         except Exception as e:
#             logging.error(f'Exception Occurred - {e}')
#             raise e

# # This class uses the above classes for data cleaning and returns the DataFrame
# class DataCleaning:
#     def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
#         self.data = data
#         self.strategy = strategy

#     def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
#         try:
#             return self.strategy.handle_data(self.data)
#         except Exception as e:
#             logging.error("Error in handling data: {}".format(e))
#             raise e








import numpy as np
import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only needs to be run once)
nltk.download('stopwords')
nltk.download('wordnet')

class DataStrategy(ABC):
    '''
        Strategy for data handling
    '''
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessing(DataStrategy):
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.scaler = StandardScaler()


    def clean_text(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        words = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words]
        return " ".join(words)

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Processing Data")
        try:
            data = data.drop_duplicates()

            # Drop unnecessary date-related columns
            date_columns = [
                "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date",
                "order_purchase_timestamp"
            ]
            data = data.drop(date_columns, axis=1, errors='ignore')

            # Drop ID columns that are not useful for modeling
            id_columns = ["customer_id", "order_id", "product_id", "seller_id","customer_unique_id","order_item_id","review_id",]
            data = data.drop(id_columns, axis=1, errors='ignore')

            # Handle missing values for numerical columns
            for column in data.columns:
                if data[column].dtype in ['float64', 'int64']:
                    median_value = data[column].median()
                    data[column].fillna(median_value, inplace=True)

            # Handle missing values for object columns
            for column in data.columns:
                if data[column].dtype == 'object':
                    data[column].fillna("Unknown", inplace=True)

            # Clean and encode the review comments
            data["review_comment_message"] = data["review_comment_message"].fillna("No review")
            data["cleaned_review_comment"] = data["review_comment_message"].apply(self.clean_text)

            bert_embeddings = self.model.encode(data["cleaned_review_comment"].tolist())
            bert_embeddings_df = pd.DataFrame(
                bert_embeddings,
                columns=[f'embedding_{i}' for i in range(bert_embeddings.shape[1])]
            )

            data = pd.concat([data.reset_index(drop=True), bert_embeddings_df], axis=1)
            data = data.drop(columns=["review_comment_message", "cleaned_review_comment"])

            # Separate the target variable
            if 'review_score' in data.columns:
                review_score = data['review_score'].reset_index(drop=True)
                data = data.drop('review_score', axis=1)
            else:
                logging.error("'review_score' column not found in data.")
                raise KeyError("'review_score' column not found in data.")

            # Identify non-numeric columns
            non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()

            # Handle categorical variables
            low_cardinality_cols = []
            high_cardinality_cols = []
            threshold = 50  # You can adjust this threshold

            for col in non_numeric_cols:
                if data[col].nunique() <= threshold:
                    low_cardinality_cols.append(col)
                else:
                    high_cardinality_cols.append(col)

            # # One-Hot Encode low cardinality columns
            # if low_cardinality_cols:
            #     one_hot_encoded = self.one_hot_encoder.fit_transform(data[low_cardinality_cols])
            #     one_hot_encoded_df = pd.DataFrame(
            #         one_hot_encoded,
            #         columns=self.one_hot_encoder.get_feature_names_out(low_cardinality_cols)
            #     )
            #     data = pd.concat([data.reset_index(drop=True), one_hot_encoded_df], axis=1)
            #     data = data.drop(low_cardinality_cols, axis=1)

            # # Label Encode high cardinality columns
            # for col in high_cardinality_cols:
            #     le = LabelEncoder()
            #     data[col] = le.fit_transform(data[col])
            #     self.label_encoders[col] = le  # Store the encoder if needed later

            # Drop any remaining non-numeric columns (if any)
            data = data.select_dtypes(include=[np.number])

            # Standardize numeric features
            numeric_cols = data.columns
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])

            # PCA to handle collinearity
            pca = PCA(n_components=0.95, random_state=42)
            pca_data = pca.fit_transform(data)
            pca_data_df = pd.DataFrame(
                pca_data,
                columns=[f'PCA_{i}' for i in range(pca_data.shape[1])]
            )

            # Add the target variable back to the DataFrame
            final_data = pd.concat([pca_data_df.reset_index(drop=True), review_score], axis=1)

            return final_data

        except Exception as e:
            logging.error(f'Exception Occurred - {e}')
            raise e


class DataDivideToTrainAndTest(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]  # target variable column
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Exception Occurred - {e}')
            raise e

# This class uses the above classes for data cleaning and returns the DataFrame
class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

