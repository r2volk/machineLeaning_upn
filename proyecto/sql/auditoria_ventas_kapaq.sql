-- Consultas de SOLO LECTURA para verificar si Kapaq tiene historial utilizable.
-- No seleccionan nombres, teléfonos, códigos de recojo ni datos de clientes.

-- 1) Cobertura de movimientos de venta disponibles.
SELECT
    COUNT(*) AS filas_movimiento_venta,
    COUNT(DISTINCT DATE(creado_en)) AS dias_con_ventas,
    COUNT(DISTINCT producto_id) AS productos_con_ventas,
    MIN(DATE(creado_en)) AS primera_fecha,
    MAX(DATE(creado_en)) AS ultima_fecha,
    SUM(ABS(cantidad)) AS unidades_vendidas
FROM movimientos_stock
WHERE tipo = 'venta';

-- 2) Resumen de devoluciones, para conocer su volumen y período.
SELECT
    COUNT(*) AS filas_movimiento_devolucion,
    COUNT(DISTINCT DATE(creado_en)) AS dias_con_devoluciones,
    COUNT(DISTINCT producto_id) AS productos_devueltos,
    MIN(DATE(creado_en)) AS primera_fecha_devolucion,
    MAX(DATE(creado_en)) AS ultima_fecha_devolucion,
    SUM(ABS(cantidad)) AS unidades_devueltas
FROM movimientos_stock
WHERE tipo = 'devolucion';

-- 3) Exportación agregada para preparar una serie diaria por producto.
-- `store = 1` identifica el único minimarket; si hay varias sucursales,
-- este valor deberá reemplazarse por el identificador correspondiente.
-- Solo se cuentan ventas brutas; las devoluciones se resumen por separado
-- en la consulta 2 para no cambiar retroactivamente la fecha de una venta.
SELECT
    DATE(creado_en) AS date,
    1 AS store,
    producto_id AS item,
    SUM(ABS(cantidad)) AS sales
FROM movimientos_stock
WHERE tipo = 'venta'
GROUP BY DATE(creado_en), producto_id
ORDER BY date, item;

-- Nota: la exportación contiene filas de días con ventas. Antes de entrenar,
-- habrá que completar con cero los días sin venta para cada producto activo.
