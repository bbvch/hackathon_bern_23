# hackathon_bern_23

[Labelstudio Instanz (labelstudiohackbern.azurewebsites.net)](labelstudiohackbern.azurewebsites.net)



# Setup on dev machine

Allow DVC on your local machine to access the azure Blob Storage:

```export AZURE_STORAGE_CONNECTION_STRING='mysecret'```

replace `mysecret` with the connection string from azure. Storage accounts > STORAGE_NAME > Access keys

Make sure you are logged into your azure account:
```az login```

