import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load your dataset
df = pd.read_csv("./data/data_geo.csv")  # Replace with your actual dataset path

# Convert order_purchase_timestamp to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# 1. Total Revenue and Monthly Revenue Trend
total_revenue = df['payment_value'].sum()
monthly_revenue = df.groupby(df['order_purchase_timestamp'].dt.to_period("M"))['payment_value'].sum().reset_index()
monthly_revenue['order_purchase_timestamp'] = monthly_revenue['order_purchase_timestamp'].astype(str)

# 2. Total Orders and Orders by Month/Season
total_orders = df['order_id'].nunique()
monthly_orders = df.groupby(df['order_purchase_timestamp'].dt.to_period("M"))['order_id'].count().reset_index()
monthly_orders['order_purchase_timestamp'] = monthly_orders['order_purchase_timestamp'].astype(str)

df['season'] = df['order_purchase_timestamp'].dt.month % 12 // 3 + 1  # Map months to seasons
seasonal_orders = df.groupby('season')['order_id'].count().reset_index()
seasonal_orders['season'] = seasonal_orders['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

# 3. Top Product Categories by Revenue
category_sales = df.groupby('product_category_name_english')['price'].sum().reset_index()
category_sales = category_sales.sort_values('price', ascending=False).head(10)

# 4. Average Order Value (AOV) on Olist and Its Variation by Product Category and Payment MethodCalculate AOV (overall)
aov = df['payment_value'].sum() / df['order_id'].nunique()

# AOV by product category
category_aov = df.groupby('product_category_name_english')['payment_value'].mean().reset_index()
category_aov = category_aov.sort_values('payment_value', ascending=False).head(10)

# AOV by payment method
payment_aov = df.groupby('payment_type')['payment_value'].mean().reset_index()

# 5. Number of Active Sellers and How It Changes Over Time
# Total active sellers
total_sellers = df['seller_id'].nunique()

# Sellers over time (monthly)
sellers_over_time = df.groupby(df['order_purchase_timestamp'].dt.to_period("M"))['seller_id'].nunique().reset_index()
sellers_over_time['order_purchase_timestamp'] = sellers_over_time['order_purchase_timestamp'].astype(str)  # Convert to string


# 6.Distribution of Seller Ratings and Their Impact on Sales Performance

# Distribution of seller ratings
rating_distribution = df['review_score'].value_counts().reset_index()
rating_distribution.columns = ['review_score', 'count']

# Sales performance by seller rating
sales_by_rating = df.groupby('review_score')['payment_value'].sum().reset_index()

# 7.Repeat Customers and Their Contribution to Sales

# Count purchases per customer
customer_purchases = df.groupby('customer_unique_id')['order_id'].count().reset_index()
customer_purchases.columns = ['customer_unique_id', 'purchase_count']

# Identify repeat customers
repeat_customers = customer_purchases[customer_purchases['purchase_count'] > 1]
num_repeat_customers = repeat_customers.shape[0]

# Total sales by repeat customers
repeat_customer_sales = df[df['customer_unique_id'].isin(repeat_customers['customer_unique_id'])]['payment_value'].sum()
total_sales = df['payment_value'].sum()

# Percentage contribution of repeat customers
repeat_sales_percentage = (repeat_customer_sales / total_sales) * 100


# 8.Average Customer Rating and Its Impact on Sales

# Average customer rating
average_rating = df['review_score'].mean()

# Sales performance by review score
sales_by_rating = df.groupby('review_score')['payment_value'].sum().reset_index()

# 9.Order Cancellation Rate and Its Impact on Seller Performance

# Calculate cancellations
total_orders = df['order_id'].nunique()
canceled_orders = df[df['order_status'] == 'canceled'].shape[0]
cancellation_rate = (canceled_orders / total_orders) * 100

# Sales lost due to cancellations
canceled_sales = df[df['order_status'] == 'canceled']['payment_value'].sum()

# 10. Top-Selling Products and Their Sales Trends Over Time
# Top-selling products by total revenue
top_products = df.groupby('product_id')['payment_value'].sum().reset_index()
top_products = top_products.sort_values('payment_value', ascending=False).head(10)

# Merge with product details for names
top_products_with_names = pd.merge(
    top_products, 
    df[['product_id', 'product_category_name_english']].drop_duplicates(), 
    on='product_id'
)

# Convert order_purchase_timestamp to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Filter for a dynamic time range (last year)
# Sales trends for top products over time
top_products_trend = df[df['product_id'].isin(top_products['product_id'])]
top_products_trend = top_products_trend.groupby([
    top_products_trend['order_purchase_timestamp'].dt.to_period("M"), 
    'product_id'
])['payment_value'].sum().reset_index()

# Merge trend data with product names
top_products_trend = pd.merge(
    top_products_trend, 
    top_products_with_names, 
    on='product_id'
)

# Correct column names
top_products_trend.rename(columns={'payment_value_x': 'payment_value'}, inplace=True)

# Convert period to string for the graph
top_products_trend['order_purchase_timestamp'] = top_products_trend['order_purchase_timestamp'].astype(str)

custom_colors = [
    '#4C6173',
    '#D68167',
    '#D9C2A7',
    '#A6705D',
    '#eee',
    '#F2D5CE',
    '#C4E1F2',
    '#359632',
    '#B38A35',

]



# 11. Most Common Payment Methods and Variations by Product Category and Geographic Region
# Payment method distribution
payment_distribution = df['payment_type'].value_counts().reset_index()
payment_distribution.columns = ['payment_type', 'count']

# Payment methods by product category
payment_by_category = df.groupby(['product_category_name_english', 'payment_type'])['payment_value'].count().reset_index()
payment_by_category = payment_by_category.sort_values('payment_value', ascending=False).head(20)

# Payment methods by geographic region
payment_by_region = df.groupby(['customer_state', 'payment_type'])['payment_value'].count().reset_index()
payment_by_region = payment_by_region.sort_values('payment_value', ascending=False).head(20)



# 12. Impact of Customer Reviews and Ratings on Sales and Product Performance
# Average rating per product
avg_rating_per_product = df.groupby('product_id')['review_score'].mean().reset_index()

# Sales performance by review score
sales_by_rating = df.groupby('review_score')['payment_value'].sum().reset_index()

# Ratings by product category
ratings_by_category = df.groupby(['product_category_name_english', 'review_score'])['payment_value'].sum().reset_index()
ratings_by_category['review_score'] = ratings_by_category['review_score'].astype(str)

rating_colors=['#DF5555','#FE8C8C','#FFB7AB','#FFD1D1','#FFFFFF','#eee']

# 13. Product Categories with the Highest Profit Margins

# Calculate profit
df['profit'] = df['price'] - df['freight_value']

# Profit margin by product category
category_profit_margin = df.groupby('product_category_name_english').agg({
    'profit': 'mean',
    'price': 'sum'
}).reset_index()
category_profit_margin['profit_margin'] = (category_profit_margin['profit'] / category_profit_margin['price']) * 100
category_profit_margin = category_profit_margin.sort_values('profit_margin', ascending=False).head(10)


# 14.Customer Retention Rate by Geolocation

# Identify repeat customers
repeat_customers = df.groupby('customer_unique_id')['order_id'].count().reset_index()
repeat_customers = repeat_customers[repeat_customers['order_id'] > 1]

# Merge repeat customers with geolocation data
df_repeat = df[df['customer_unique_id'].isin(repeat_customers['customer_unique_id'])]
retention_by_geolocation = df_repeat.groupby('geolocation_city')['customer_unique_id'].nunique().reset_index()
total_customers_by_geolocation = df.groupby('geolocation_city')['customer_unique_id'].nunique().reset_index()
total_customers_by_geolocation.columns = ['geolocation_city', 'total_customers']

# Calculate retention rate
retention_rate = pd.merge(retention_by_geolocation, total_customers_by_geolocation, on='geolocation_city')
retention_rate['retention_rate'] = (retention_rate['customer_unique_id'] / retention_rate['total_customers']) * 100
retention_rate = retention_rate.sort_values('retention_rate', ascending=False).head(10)



# Profit and Revenue by Geolocation

# Calculate profit and revenue for each city
geolocation_metrics = df.groupby(['geolocation_city', 'geolocation_lat', 'geolocation_lng']).agg({
    'profit': 'sum',
    'price': 'sum'
}).reset_index()
geolocation_metrics.columns = ['city', 'lat', 'lng', 'total_profit', 'total_revenue']


# Customer Retention by Geolocation (Map)
# Identify repeat customers

# Retention rate by geolocation
retention_metrics = df_repeat.groupby(['geolocation_city', 'geolocation_lat', 'geolocation_lng']).agg({
    'customer_unique_id': 'nunique'
}).reset_index()
total_customers_by_geolocation = df.groupby(['geolocation_city', 'geolocation_lat', 'geolocation_lng']).agg({
    'customer_unique_id': 'nunique'
}).reset_index()
retention_rate_map = pd.merge(retention_metrics, total_customers_by_geolocation, on=['geolocation_city', 'geolocation_lat', 'geolocation_lng'])
retention_rate_map['retention_rate'] = (retention_rate_map['customer_unique_id_x'] / retention_rate_map['customer_unique_id_y']) * 100
retention_rate_map.columns = ['city', 'lat', 'lng', 'repeat_customers', 'total_customers', 'retention_rate']



# . Revenue by State and Customer Density
# Revenue by state
revenue_by_state = df.groupby('customer_state')['payment_value'].sum().reset_index()
revenue_by_state = revenue_by_state.sort_values('payment_value', ascending=False)

# Add "Others" category for states outside the top 5
top_5_states = revenue_by_state.head(5)
others = pd.DataFrame({
    'customer_state': ['Others'],
    'payment_value': [revenue_by_state.iloc[5:]['payment_value'].sum()]
})
revenue_with_others = pd.concat([top_5_states, others], ignore_index=True)



df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Calculate delivery time
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Delivery time by state

delivery_time_by_state = df.groupby('customer_state')['delivery_time'].mean().reset_index()
delivery_time_by_state = delivery_time_by_state.sort_values('delivery_time', ascending=True)
avg_delivery_time= df['delivery_time'].mean()

# Delivery time by product category
delivery_time_by_category = df.groupby('product_category_name_english')['delivery_time'].mean().reset_index()
delivery_time_by_category = delivery_time_by_category.sort_values('delivery_time', ascending=True).head(20)
# Initialize Dash app
app = dash.Dash(__name__)

# Use dark theme
dark_theme_style = {
    'backgroundColor': '#1E1E1E',
    'color': '#eee',
}

# Layout
app.layout = html.Div(style=dark_theme_style, children=[
    html.H1("E-Commerce Dashboard", style={'textAlign': 'center', 'color': '#eee', 'padding': '15px'}),

    # Total Revenue and Monthly Revenue Trend
    html.Div([
        html.H2(f"Total Revenue: ${total_revenue:,.2f}", style={'color': '#eee', 'padding': '15px'}),
        dcc.Graph(
            figure=px.line(
                monthly_revenue,
                x='order_purchase_timestamp',
                y='payment_value',
                labels={'order_purchase_timestamp': 'Month', 'payment_value': 'Revenue ($)'},
                title="Monthly Revenue Trend",
                line_shape='linear'
            ).update_traces(line_color='#049DBF')
            .update_layout(template="plotly_dark")
        )

    ], style={'width': '100%', 'display': 'block'}),

    # Orders by Month and Season 
    html.Div([
        html.Div([
            html.H2(f"Total Orders: {total_orders:,}", style={'color': '#eee', 'padding': '10px'}),
            dcc.Graph(
            figure=px.bar(
                monthly_orders,
                x='order_purchase_timestamp',
                y='order_id',
                labels={'order_purchase_timestamp': 'Month', 'order_id': 'Number of Orders'},
                title="Monthly Order Trends",
                color_discrete_sequence=['#D98E7E']  
            ).update_layout(template="plotly_dark")
        )
        ], style={'width': '48%', 'padding': '10px'}),

        html.Div([
            dcc.Graph(
                figure=px.bar(
                    seasonal_orders,
                    x='season',
                    y='order_id',
                    labels={'season': 'Season', 'order_id': 'Number of Orders'},
                    title="Orders by Season",
                    color_discrete_sequence=['#04C4D9']
                ).update_layout(template="plotly_dark")
            )
        ], style={'width': '48%', 'padding': '10px','margin-top':'80px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}
    ),

    # Top Product Categories by Revenue
    html.Div([
        html.H2("Top Product Categories by Revenue", style={'color': '#eee', 'padding': '15px'}),
        dcc.Graph(
            figure=px.bar(
                category_sales,
                x='product_category_name_english',
                y='price',
                labels={'product_category_name_english': 'Category', 'price': 'Revenue ($)'},
                title="Top Product Categories",
                color_discrete_sequence=['#F2C1B6']
            ).update_layout(template="plotly_dark",xaxis_tickangle=30 )
        )
    ], style={'width': '100%', 'padding': '10px'}),
    # Average Order Value (AOV) 
    html.Div([
        html.Div([
        html.H2(f"Average Order Value (AOV): ${aov:.2f}", style={'color': '#eee', 'padding': '15px'}),
        dcc.Graph(
            figure=px.bar(
                category_aov,
                x='product_category_name_english',
                y='payment_value',
                labels={'product_category_name_english': 'Category', 'payment_value': 'AOV ($)'},
                title="AOV by Product Category",
                color_discrete_sequence=['#D6D6C9']
            ).update_layout(template="plotly_dark")
        ),
        ], style={'width': '48%', 'padding': '10px'}),
        html.Div([
        dcc.Graph(
            figure=px.bar(
                payment_aov,
                x='payment_type',
                y='payment_value',
                labels={'payment_type': 'Payment Method', 'payment_value': 'AOV ($)'},
                title="AOV by Payment Method",
                color_discrete_sequence=['#D2D99C']
            ).update_layout(template="plotly_dark")
        )
    ], style={'width': '48%', 'padding': '10px','margin-top':'95px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}
        ),
    # Number of Active Sellers and How It Changes Over Time
    html.Div([
        html.H2(f"Total Active Sellers: {total_sellers:,}",style={'color': '#eee', 'padding': '15px'}),
        dcc.Graph(
            figure=px.line(
                sellers_over_time,
                x='order_purchase_timestamp',
                y='seller_id',
                labels={'order_purchase_timestamp': 'Month', 'seller_id': 'Active Sellers'},
                title="Active Sellers Over Time"
            ).update_traces(line_color='#D68167').update_layout(template="plotly_dark")
        )
    ]),
    # Distribution of Seller Ratings and Their Impact on Sales Performance
    html.Div([
        
        html.Div([
            html.H2("Seller Rating Distribution",
                style={
                'color': '#eee',
                'padding': '15px',
            }),
            html.P(f"\nAverage Customer Rating: {average_rating:.2f}",style={
                'color': '#eee',
                'padding-left': '15px',
            }),
            dcc.Graph(
                figure=px.bar(
                    rating_distribution,
                    x='review_score',
                    y='count',
                    labels={'review_score': 'Rating', 'count': 'Number of Reviews'},
                    title="Seller Rating Distribution",
                    color_discrete_sequence=['#FFFAD5']
                ).update_layout(template="plotly_dark")
            ),
        ],style={'width': '48%', }),
        html.Div([
            dcc.Graph(
                figure=px.bar(
                    sales_by_rating,
                    x='review_score',
                    y='payment_value',
                    labels={'review_score': 'Rating', 'payment_value': 'Total Sales ($)'},
                    title="Sales Performance by Seller Rating",
                    color_discrete_sequence=['#1c9fad']
                ).update_layout(template="plotly_dark")
            )
        ],style={'width': '48%', 'padding': '10px','margin-top':'100px'}),
    ],style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}
    ),

    html.Div([
        html.H2("Impact of Ratings on Sales", 
                style={
                    'color': '#eee',
                    'padding': '15px',
                }),
        dcc.Graph(
            figure=px.bar(
                ratings_by_category,
                x='product_category_name_english',
                y='payment_value',
                color='review_score',
                labels={'product_category_name_english': 'Category', 'payment_value': 'Revenue ($)', 'review_score': 'Rating'},
                title="Sales by Rating and Product Category",
                color_discrete_sequence=rating_colors 
            ).update_layout(template="plotly_dark",height=600 )
        )
    ]),
    # Repeat Customers and Their Contribution to Sales
    html.Div([
        html.H2(
            f"Repeat Customers: {num_repeat_customers:,}",
            style={
                'color': '#eee',
                'textAlign': 'center'
            }
        ),
        html.H3(
            f"Percentage of Sales from Repeat Customers: {repeat_sales_percentage:.2f}%",
            style={
                'color': '#eee',
                'padding': '10px',
                'textAlign': 'center'
            }
        ),
        dcc.Graph(
            figure=px.bar(
                x=['Repeat Customers', 'Other Customers'],
                y=[repeat_customer_sales, total_sales - repeat_customer_sales],
                labels={'x': 'Customer Type', 'y': 'Sales ($)'},
                title="Sales Contribution by Repeat Customers",
                color_discrete_sequence=['#F2E0C9']
            ).update_layout(
                template="plotly_dark"
            )
        )
    ]),
    html.Div([
        html.H2("Customer Retention by Geolocation",style={
                'color': '#eee',
                'padding': '15px',
            }),
        dcc.Graph(
            figure=px.scatter_mapbox(
                retention_rate_map,
                lat='lat',
                lon='lng',
                size='retention_rate',
                color='retention_rate',
                hover_name='city',
                hover_data={'repeat_customers': True, 'total_customers': True, 'retention_rate': True},
                labels={'retention_rate': 'Retention Rate (%)'},
                title="Customer Retention by City",
                color_continuous_scale='Blues',
                zoom=4
            ).update_layout(template="plotly_dark",mapbox_style="carto-darkmatter",height=700 )
        )
    ]),
    # Order Cancellation Rate and Its Impact on Seller Performance
    html.Div([
        html.H2(f"Order Cancellation Rate: {cancellation_rate:.2f}%",
            style={
                'color': '#eee',
                'padding': '10px',
                'textAlign': 'center'
            }),
        html.H3(f"Sales Lost Due to Cancellations: ${canceled_sales:,.2f}",
                style={
                'color': '#eee',
                'textAlign': 'center'
            }),
        dcc.Graph(
            figure=px.bar(
                x=['Canceled Orders', 'Completed Orders'],
                y=[canceled_orders, total_orders - canceled_orders],
                labels={'x': 'Order Status', 'y': 'Number of Orders'},
                title="Order Cancellation vs Completion",
                color_discrete_sequence=['#BD4932']
            ).update_layout(
                template="plotly_dark"
            )
        )
    ]),

    html.Div([
        html.H2("Top-Selling Products by Revenue",
                style={
                'color': '#eee',
                'padding': '15px',
            }),
        dcc.Graph(
            figure=px.bar(
                top_products_with_names,
                x='product_category_name_english',
                y='payment_value',
                labels={'product_category_name_english': 'Product Category', 'payment_value': 'Revenue ($)'},
                title="Top-Selling Products",
                color_discrete_sequence=['#D9C2A7']
            ).update_layout(
                template="plotly_dark"
            )
        ),
        dcc.Graph(
            figure=px.line(
                top_products_trend,
                x='order_purchase_timestamp',
                y='payment_value',
                color='product_category_name_english',
                labels={
                    'order_purchase_timestamp': 'Month',
                    'payment_value': 'Revenue ($)',
                    'product_category_name_english': 'Product Name'
                },
                title="Sales Trends for Top-Selling Products",
                hover_data={
                    'payment_value': ':.2f',  # Format revenue to 2 decimal places
                    'product_category_name_english': True,
                    'order_purchase_timestamp': True
                },
                color_discrete_sequence=custom_colors  
            ).update_layout(template="plotly_dark")
            )
    ]),

        html.Div([
            html.H2("Most Common Payment Methods"
                    , style={
                    'color': '#eee',
                    'padding': '15px',
                }),
            dcc.Graph(
                figure=px.bar(
                    payment_distribution,
                    x='payment_type',
                    y='count',
                    labels={'payment_type': 'Payment Method', 'count': 'Count'},
                    title="Payment Method Distribution",
                    color_discrete_sequence=['#5AADBF']
                ).update_layout(template="plotly_dark")
            ),
        ]),

        html.Div([
            html.Div([
                dcc.Graph(
                    figure=px.bar(
                        payment_by_category,
                        x='product_category_name_english',
                        y='payment_value',
                        color='payment_type',
                        labels={'product_category_name_english': 'Product Category', 'payment_value': 'Count', 'payment_type': 'Payment Method'},
                        title="Payment Methods by Product Category",
                        color_discrete_sequence=custom_colors
                    ).update_layout(template="plotly_dark")
                ),
            ],style={'width': '48%', 'padding': '10px'}),

            html.Div([
                dcc.Graph(
                    figure=px.bar(
                        payment_by_region,
                        x='customer_state',
                        y='payment_value',
                        color='payment_type',
                        labels={'customer_state': 'State', 'payment_value': 'Count', 'payment_type': 'Payment Method'},
                        title="Payment Methods by Geographic Region",
                        color_discrete_sequence=custom_colors  
                    ).update_layout(template="plotly_dark")
                )
        ],style={'width': '48%', 'padding': '10px'}),
    ],style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),

    html.Div([
        html.H2("Product Categories with the Highest Profit Margins", 
                style={
                    'color': '#eee',
                    'padding': '15px',
                }),
        dcc.Graph(
            figure=px.bar(
                category_profit_margin,
                x='product_category_name_english',
                y='profit_margin',
                labels={'product_category_name_english': 'Category', 'profit_margin': 'Profit Margin (%)'},
                title="Top Categories by Profit Margin",
                color_discrete_sequence=['#DF5062']
            ).update_layout(template="plotly_dark")
        )
    ]),
    html.Div([
        html.H2("Profit and Revenue by Geolocation",
                style={
                    'color': '#eee',
                    'padding': '15px',
                }),
        dcc.Graph(
            figure=px.scatter_mapbox(
                geolocation_metrics,
                lat='lat',
                lon='lng',
                size='total_revenue',
                color='total_profit',
                hover_name='city',
                hover_data={'total_profit': True, 'total_revenue': True},
                labels={'total_profit': 'Profit ($)', 'total_revenue': 'Revenue ($)'},
                title="Profit and Revenue by City",
                color_continuous_scale=rating_colors,
                zoom=4
            ).update_layout(template="plotly_dark",mapbox_style="carto-darkmatter",height=700 )
        )
    ]),

    html.Div([
        html.Div([
            dcc.Graph(
                figure=px.bar(
                    revenue_by_state,
                    x='customer_state',
                    y='payment_value',
                    labels={'customer_state': 'State', 'payment_value': 'Total Revenue ($)'},
                    title="Total Revenue by State",
                    color_discrete_sequence=['#008F8C']
                ).update_layout(template="plotly_dark")
            )
        ],style={'width': '48%', 'padding': '10px'}),

        html.Div([
            dcc.Graph(
                figure=px.pie(
                revenue_with_others,
                names='customer_state',
                values='payment_value',
                title="Revenue Distribution by State (Top 5 and Others)",
                labels={'customer_state': 'State', 'payment_value': 'Revenue ($)'}
            ).update_layout(template="plotly_dark").update_traces(textinfo='percent+label',marker=dict(colors=custom_colors))
        )
        ],style={'width': '48%', 'padding': '10px'}),

    ],style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}
    ),

    html.Div([
        html.Div([
                html.H2(f"Average Delivery Time {avg_delivery_time:,.2f} days",
                style={
                    'color': '#eee',
                    'padding': '15px',
                }),

            dcc.Graph(
                figure=px.bar(
                delivery_time_by_state,
                x='customer_state',
                y='delivery_time',
                labels={'customer_state': 'State', 'delivery_time': 'Average Delivery Time (days)'},
                title="Average Delivery Time by State",
                color_discrete_sequence=['#04BF9D']

            ).update_layout(template="plotly_dark")
            )
        ],style={'width': '48%', 'padding': '10px'}),

        html.Div([
            dcc.Graph(
                figure=px.bar(
                delivery_time_by_category,
                x='product_category_name_english',
                y='delivery_time',
                labels={'product_category_name_english': 'Category', 'delivery_time': 'Average Delivery Time (days)'},
                title="Average Delivery Time by Category",
                color_discrete_sequence=['#BF665E']

            ).update_layout(template="plotly_dark",height=550)
                )
        ],style={'width': '48%', 'padding': '10px'}),

    ],style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}
    ),



])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
