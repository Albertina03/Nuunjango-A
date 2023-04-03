
using CSV
using DataFrames
using Dates
sale = CSV.read("sales-of-shampoo-over-a-three-ye.csv",DataFrame,header=true)
print(sale)


# add years to month column
for i in 1:length(sale.Month)

    if occursin("1-", sale.Month[i])
        sale.Month[i] = "2019-" * sale.Month[i]
    elseif occursin("2-", sale.Month[i])
        sale.Month[i] = "2020-" * sale.Month[i]
    elseif occursin("3-", sale.Month[i])
        sale.Month[i] = "2021-" * sale.Month[i]
    end
end
# print the updated dataframe
println(sale)
describe(sale)

using Plots
plot(sale."Sales of shampoo over a three year period", bins=10, xlabel="Sales of shampoo over a three year period", ylabel="Sales of shampoo", label="Sales over Time",title="Distribution of Sales")

# +
using Plots


# Group the data by year
data_2019 = filter(row -> occursin("2019", row.Month), sale)
data_2020 = filter(row -> occursin("2020", row.Month), sale)
data_2021 = filter(row -> occursin("2021", row.Month), sale)

# calculate total sales for each year
sales_2019 = sum(data_2019."Sales of shampoo over a three year period")
sales_2020 = sum(data_2020."Sales of shampoo over a three year period")
sales_2021 = sum(data_2021."Sales of shampoo over a three year period")

# Create a line plot of the data
plot(["2019","2020","2022"] , [sales_2019, sales_2020, sales_2021], 
    label="Sales over Time",
    xlabel="Year",
    ylabel="Sales of shampoo ",
    legend=:topleft,
    title="Sales of Shampoo by Year")





using Random
# Split the dataframe into training and testing sets
Random.seed!(123) # for reproducibility
sale_shuffled = sale[shuffle(1:end), :]

train_ratio = 0.7
train_size = Int(floor(size(sale, 1) * train_ratio))
train_data = sale_shuffled[1:train_size, :]
test_data = sale_shuffled[train_size+1:end, :]


# View the resulting train and test sets
println("Training set:")
display(train_data)
println("Test set:")
display(test_data)




using Pkg
Pkg.add("MLJ")
Pkg.add("GLM")
using MLJ
using GLM

# Create the linear regression model
model=lm(@formula("Sales of shampoo over a three year period" ~ Month),train_data)

# Fit model to training data
model = machine(model, sale, target="Sales of shampoo over a three year period")
fit!(model,train_data)

# Make predictions on test data
pred= predict(model, test_data)

# Calculate model performance on test data

mse =mean_squared_error(pred, sale[test_data,"Sales of shampoo over a three year period" ])


using Flux

# Define the model
model = Chain(Dense(1, 1))

# Define the loss function
loss(x, y) = Flux.mse(model(x), y)


using Flux.Optimise

# Prepare the data for training
x_train = [i for i in 1:size(train_data, 1)]
y_train = train_data[:, 2]

# Train the model
sale = [(x_train[i], y_train[i]) for i in 1:size(train_data, 1)]
opt_data = ADAM(0.1)
Flux.train!(loss, params(model), sale, opt_data)

# Prepare the data for testing
x_test = [i for i in 1:size(test_data, 1)]
y_test = test_data[:, 2]

# Use the model to make predictions on the test data
y_pred = model(x_test)

# Calculate the mean squared error
mse = Flux.mse(y_pred, y_test)

# Prepare the new data for prediction
x_data = [size(sale, 1)+1, size(sale, 1)+2, size(sale, 1)+3]

# Use the model to make predictions on the new data
y_pred_new = model(x_data)





