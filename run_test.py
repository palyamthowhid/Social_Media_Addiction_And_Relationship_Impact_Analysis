from app import app

with app.test_client() as client:
    data={
        'Age':'21','Gender':'Male','Academic_Level':'Undergraduate','Country':'India',
        'Avg_Daily_Usage_Hours':'4','Most_Used_Platform':'Facebook','Affects_Academic_Performance':'Yes',
        'Sleep_Hours_Per_Night':'6','Relationship_Status':'Single','Conflicts_Over_Social_Media':'2',
        'Self_Perceived_Addiction':'5'
    }
    rv=client.post('/predict', data=data)
    print('status', rv.status_code)
    print(rv.data.decode('utf-8')[:1000])
