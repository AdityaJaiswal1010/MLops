import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessing
def get_data_for_test():
    try:
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
        # logging.info(merged_data)
        # date_columns = [
        #         "order_approved_at", "order_delivered_carrier_date",
        #         "order_delivered_customer_date", "order_estimated_delivery_date",
        #         "order_purchase_timestamp","customer_id", "order_id", "product_id", "seller_id",
        #     ]
        # merged_data = merged_data.drop(date_columns, axis=1, errors='ignore')
        df=merged_data
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessing()
        data_cleaning = DataCleaning (df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e