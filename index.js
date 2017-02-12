#!/usr/local/bin/node

// index.js
// 

var request  = require("request")
var PersonalityInsightsV3 = require('watson-developer-cloud/personality-insights/v3'
  )

// initialize personality insight object
var personality_insights = new PersonalityInsightsV3({
  username: '357be82f-1ed1-4420-a9ae-d2b97ad4d1a4',
  password: 'f6LfKRzNBwH3',
  version_date: '2016-10-20'
});

// set input json tweet files
if (process.argv[2] !== undefined) {
  var profile = './'+process.argv[2]+'.json'
}
else {
  var profile = './profile.json'
}

// params for request to IBM personality
var params = {
  // Get the content items from the JSON file.
  content_items: require(profile).contentItems,
  consumption_preferences: true,
  raw_scores: true,
  headers: {
    'accept-language': 'en',
    'accept': 'application/json'
  }
}

personality_insights.profile(params, function(error, response) {
  if (error)
    console.log('Error:', error);
  else
    console.log(JSON.stringify(response, null, 2));
  }
);