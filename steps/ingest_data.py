import logging
import pandas as pd
from zenml import step

class IngestData:    
    def get_data(self):
        logging.info(f"Ingesting data")
        # as there are 9 datset inter related so we need to merge them depending upon the common columns
        # Load each dataset
        orders = pd.read_csv('data/olist_orders_dataset.csv')
        customers = pd.read_csv('data/olist_customers_dataset.csv')
        order_items = pd.read_csv('data/olist_order_items_dataset.csv')
        order_payments = pd.read_csv('data/olist_order_payments_dataset.csv')
        order_reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
        products = pd.read_csv('data/olist_products_dataset.csv')
        sellers = pd.read_csv('data/olist_sellers_dataset.csv')
        geolocation = pd.read_csv('data/olist_geolocation_dataset.csv')
        category_translation = pd.read_csv('data/product_category_name_translation.csv')


        merged_data = orders.copy()
        merged_data = merged_data.merge(customers, on='customer_id', how='left')
        merged_data = merged_data.merge(order_items, on='order_id', how='left')
        merged_data = merged_data.merge(products, on='product_id', how='left')
        merged_data = merged_data.merge(order_payments, on='order_id', how='left')
        merged_data = merged_data.merge(order_reviews, on='order_id', how='left')
        merged_data = merged_data.merge(sellers, on='seller_id', how='left')
        merged_data = merged_data.merge(category_translation, on='product_category_name', how='left')
        # Note: geolocation dataset doesn't directly join to other tables, but you might aggregate or merge it based on customer locations
        logging.info(merged_data)
        date_columns = [
                "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date",
                "order_purchase_timestamp","customer_id", "order_id", "product_id", "seller_id",
            ]
        merged_data = merged_data.drop(date_columns, axis=1, errors='ignore')

        # Save the DataFrame to a CSV file in the specified folder, overwriting each time
        merged_data.to_csv('./data/merged_data.csv', index=False)
        logging.info(merged_data.info())
        logging.info(merged_data.describe())
        return merged_data
@step
def ingest_data() -> pd.DataFrame:
    try:
        return IngestData().get_data()
    except Exception as e:
        logging.info("Error while ingesting the data - {e}")
        raise e