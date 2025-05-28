const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

// Serve static files from public directory
app.use(express.static('public'));

// Serve JSON files
app.use('/json', express.static('json'));

// Start server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});