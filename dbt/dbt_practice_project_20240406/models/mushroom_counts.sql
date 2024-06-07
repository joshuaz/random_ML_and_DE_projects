SELECT
  cap_shape,
  COUNT(*) AS count
FROM
  {{ ref('mushroom_cleaned') }}
GROUP BY
  cap_shape;
