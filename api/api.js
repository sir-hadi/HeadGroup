const express = require('express');
const pool = require('./db');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/sl_vision_post', async (req, res) => {
  try {
    const { id_cam, id_dot, people_count } = req.query;
    console.log(id_cam, id_dot, people_count)
    const newVision = await pool.query(
      'INSERT INTO sl_vision (id_cam,id_dot,people_count,time_vision) VALUES ($1,$2,$3, to_timestamp($4 / 1000.0)) RETURNING *',
      [id_cam, id_dot, people_count, Date.now()]
    );
    // res.json(newVision.rows[0]);
    res.sendStatus(200);
  } catch (err) {
    console.log(err.message);
  }
});

app.get('/sl_vision_last_count', async (req, res) => {
  const { id_cam, id_dot } = req.query;
  const lastVision = await pool.query(
    'SELECT * FROM sl_vision WHERE id=(SELECT max(id) FROM sl_vision WHERE (id_cam = ' +
      id_cam +
      ') AND (id_dot = ' +
      id_dot +
      '));'
  );
  var data = lastVision.rows[0];
  console.log(data);
  resData = {
    people_count: data.people_count,
  };

  res.json(resData);
});

app.get('/sl_vision_get_all', async (req, res) => {
    const lastVision = await pool.query(
      'SELECT * FROM sl_vision;'
    );
    var data = lastVision.rows[0];
    console.log(data);
    res.json(data);
  });

app.listen(5050, () => console.log(`Smartbeauty Backend API server started!`));
