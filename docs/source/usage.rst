Getting Started
=====


.. _prerequisites:

Prerequisites
------------
1. GNU `make` utility (`link <https://www.gnu.org/software/make/>`_)
2. Python of version 3.7.13 (`link <https://www.python.org/downloads/release/python-3713/>`_)
3. Packaging manager `poetry` (`link <https://python-poetry.org>``)
4. At least 2Gb on your hard disk

.. code-block:: console

   poetry lock
   poetry --no-root install

Run application locally
----------------

To your delight, it's done via a single command:

.. code-block:: console

   poetry run make build
