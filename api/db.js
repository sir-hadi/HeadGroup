const Pool = require("pg").Pool;

const pool = new Pool({
    user: "smartlight",
    password: "lampupintar",
    database: "smartlight_db",
    host: "localhost",
    port: 5432
})

module.exports = pool;  